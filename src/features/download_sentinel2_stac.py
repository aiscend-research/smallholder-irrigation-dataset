#!/usr/bin/env python3
"""
Sentinel-2 time-series downloader using a public STAC catalog (Microsoft
Planetary Computer) instead of Google Earth Engine.

Why this exists
---------------
The original `download_sentinel2.py` requires a Google Earth Engine service
account (gated by an application process). This module is a drop-in
replacement that needs no signup: it queries the Microsoft Planetary Computer
STAC catalog for the `sentinel-2-l2a` collection, signs asset URLs anonymously,
and reads windowed bytes from cloud-optimized GeoTIFFs directly.

What it produces
----------------
Identical outputs to the GEE script — same filenames, same metadata schema,
same band order, same 100x100 @ 10 m grid — so all downstream code
(`create_label_band.py`, the modeling pipeline) just works.

How it differs from the GEE path
--------------------------------
- L2A only. MPC's L2A coverage extends to 2016 globally, so L1C isn't needed
  for this project.
- Scene ranking uses the STAC `eo:cloud_cover` property (per-scene %) rather
  than counting good pixels in the AOI. This is faster (no extra SCL read per
  candidate) and good enough for a 1 km AOI; the masked stack still uses the
  full SCL-based mask once the best scene is picked.
- Cloud masking keeps the same SCL classes as the GEE version: 4 (vegetation),
  5 (bare soil), 6 (water), 7 (unclassified), 11 (snow).

Usage
-----
    python src/features/download_sentinel2_stac.py
"""

import json
import logging
import math
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd
import pystac_client
import planetary_computer
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform as warp_transform
from rasterio.windows import from_bounds

# Project setup (matches download_sentinel2.py convention)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.features.download_sentinel2 import (
    S2_BANDS,
    dataset_download,
    get_stats,
    retrieve_time_series_stack,
)
from src.utils.utils import get_data_root

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION_L2A = "sentinel-2-l2a"

# SCL classes considered "good" (matches GEE script's get_quality_mask L2A)
GOOD_SCL_CLASSES = {4, 5, 6, 7, 11}

# GEE uses 'B2'..'B12'; MPC asset keys are zero-padded ('B02'..'B12', 'B8A').
_MPC_KEY = {
    "B2": "B02", "B3": "B03", "B4": "B04", "B5": "B05",
    "B6": "B06", "B7": "B07", "B8": "B08", "B8A": "B8A",
    "B11": "B11", "B12": "B12",
}

# AOI half-side in meters (GEE script uses `point.buffer(500).bounds()` → ~1 km box)
AOI_HALF_M = 500


# ---------------------------------------------------------------------------
# STAC catalog (single shared client, signs URLs anonymously)
# ---------------------------------------------------------------------------
_catalog = None


def _get_catalog():
    """Lazy global STAC client. `sign_inplace` adds short-lived SAS tokens to
    all asset hrefs returned by searches — no Azure/Microsoft account needed."""
    global _catalog
    if _catalog is None:
        _catalog = pystac_client.Client.open(
            STAC_URL,
            modifier=planetary_computer.sign_inplace,
        )
    return _catalog


# ---------------------------------------------------------------------------
# AOI math
# ---------------------------------------------------------------------------
def _aoi_bounds_utm(lat: float, lon: float, dst_crs) -> tuple:
    """Return a ~1 km x 1 km bounding box around (lat, lon) in the target CRS.

    GEE's `point.buffer(500).bounds()` produces a 1000 m square centered on
    the point in a local projected CRS. We do the same: project the point to
    the scene's UTM CRS, then offset by ±500 m on each axis.
    """
    xs, ys = warp_transform("EPSG:4326", dst_crs, [lon], [lat])
    cx, cy = xs[0], ys[0]
    return (cx - AOI_HALF_M, cy - AOI_HALF_M, cx + AOI_HALF_M, cy + AOI_HALF_M)


# ---------------------------------------------------------------------------
# Scene selection
# ---------------------------------------------------------------------------
def _search_best_item(lat: float, lon: float, start_date: str, end_date: str):
    """Find the lowest-cloud-cover scene that intersects the point in the window."""
    cat = _get_catalog()
    search = cat.search(
        collections=[COLLECTION_L2A],
        intersects={"type": "Point", "coordinates": [lon, lat]},
        datetime=f"{start_date}/{end_date}",
    )
    items = list(search.items())
    if not items:
        return None

    def cc(item):
        v = item.properties.get("eo:cloud_cover")
        return v if v is not None else math.inf

    items.sort(key=cc)
    return items[0]


# ---------------------------------------------------------------------------
# Windowed COG reads
# ---------------------------------------------------------------------------
def _read_band_to_grid(href: str, bounds_utm, target_size: int, resampling: Resampling):
    """Open a COG via signed HTTPS, read just the AOI window, and resample to
    `target_size x target_size`. Returns (array, transform, crs)."""
    with rasterio.open(href) as src:
        win = from_bounds(*bounds_utm, transform=src.transform)
        arr = src.read(
            1,
            window=win,
            out_shape=(target_size, target_size),
            resampling=resampling,
            boundless=True,
            fill_value=0,
        )
        # Build a transform aligned to the requested 10 m grid.
        win_transform = src.window_transform(win)
        scale_x = win.width / target_size
        scale_y = win.height / target_size
        out_transform = win_transform * rasterio.Affine.scale(scale_x, scale_y)
        return arr.astype(np.uint16), out_transform, src.crs


# ---------------------------------------------------------------------------
# GeoTIFF writer (matches GEE script's output: uint16, nodata=0)
# ---------------------------------------------------------------------------
def _write_stack(path: str, stack: np.ndarray, transform, crs):
    bands, h, w = stack.shape
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=h, width=w,
        count=bands,
        dtype="uint16",
        crs=crs,
        transform=transform,
        nodata=0,
        compress="deflate",
    ) as dst:
        dst.write(stack)


# ---------------------------------------------------------------------------
# Public exporter: drop-in replacement for download_sentinel2.s2_image_exporter
# ---------------------------------------------------------------------------
def s2_image_exporter_stac(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    file_name: str,
    out_dir: str,
    collection: str = "L2A",  # accepted for API parity; ignored (always L2A)
) -> bool:
    """
    Download best-quality Sentinel-2 L2A scene for a time window via MPC STAC.

    Mirrors the signature of `download_sentinel2.s2_image_exporter` so it can
    be passed as `scene_exporter` to `retrieve_time_series_stack` /
    `dataset_download`.

    Writes:
        {out_dir}/{file_name}.tif         — unmasked 10-band stack (100x100)
        {out_dir}/{file_name}_masked.tif  — same, with non-good SCL pixels = 0

    Returns True on success, False if no scene found (or on any read failure).
    """
    if collection != "L2A":
        # The GEE script defaults to L1C; we silently use L2A regardless.
        # MPC L2A covers the full 2016+ archive in Africa, so this loses nothing
        # for the Zambia dataset.
        logging.debug(f"STAC backend always uses L2A (requested {collection})")

    os.makedirs(out_dir, exist_ok=True)

    item = _search_best_item(lat, lon, start_date, end_date)
    if item is None:
        logging.warning(f"No L2A scenes for ({lat:.4f},{lon:.4f}) {start_date}..{end_date}")
        return False

    target_size = 100  # 1 km AOI at 10 m

    # We don't know the scene CRS until we open the first asset, so compute
    # the bounds against B02 (10 m, always present, fast metadata read).
    b02_href = item.assets["B02"].href
    with rasterio.open(b02_href) as src:
        scene_crs = src.crs
    bounds_utm = _aoi_bounds_utm(lat, lon, scene_crs)

    # Read each band; native resolutions: B02/B03/B04/B08 are 10 m, the rest 20 m.
    band_arrays = []
    out_transform = None
    out_crs = None
    try:
        for band in S2_BANDS:
            asset = item.assets[_MPC_KEY[band]]
            arr, tfm, crs = _read_band_to_grid(
                asset.href, bounds_utm, target_size, Resampling.bilinear
            )
            band_arrays.append(arr)
            if out_transform is None:
                out_transform = tfm
                out_crs = crs

        # SCL for masking (20 m → upsample with nearest to preserve class labels)
        scl_arr, _, _ = _read_band_to_grid(
            item.assets["SCL"].href, bounds_utm, target_size, Resampling.nearest
        )
    except Exception as e:
        logging.error(f"Failed reading {item.id}: {e}")
        return False

    unmasked = np.stack(band_arrays, axis=0)  # (10, 100, 100)

    # SCL-based quality mask: 1 = keep, 0 = bad
    good = np.isin(scl_arr, list(GOOD_SCL_CLASSES))
    masked = unmasked * good[np.newaxis, :, :].astype(np.uint16)

    _write_stack(os.path.join(out_dir, f"{file_name}.tif"),
                 unmasked, out_transform, out_crs)
    _write_stack(os.path.join(out_dir, f"{file_name}_masked.tif"),
                 masked, out_transform, out_crs)
    return True


# ---------------------------------------------------------------------------
# Site-level parallel runner
# ---------------------------------------------------------------------------
def dataset_download_parallel(
    csv: str,
    download_dir: str,
    max_concurrent_sites: int = 6,
    subset: bool = False,
    resume_dir: str = None,
    start_month: int = 1,
    num_windows: int = 36,
    timestep: int = 10,
    window_buffer: int = 3,
    target_size: int = 100,
):
    """Like dataset_download() but processes N sites in parallel.

    Each site already uses 10 internal threads for per-window fetches; with
    max_concurrent_sites=6 we'll keep up to ~60 HTTPS connections open at once,
    which MPC tolerates comfortably. Resume capability (skip-if-stack-exists)
    is preserved.
    """
    # Version / resume folder
    if resume_dir:
        out_dir = os.path.join(download_dir, resume_dir)
        if not os.path.exists(out_dir):
            raise ValueError(f"Resume directory does not exist: {out_dir}")
        logging.info(f"Resuming into existing directory: {out_dir}")
    else:
        version_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(download_dir, version_name)
        os.makedirs(out_dir, exist_ok=False)
        run_meta = {
            "csv": csv,
            "collection": "L2A",
            "start_month": start_month,
            "num_windows": num_windows,
            "timestep": timestep,
            "window_buffer": window_buffer,
            "target_size": target_size,
            "subset": subset,
            "max_concurrent_sites": max_concurrent_sites,
            "scene_exporter": "s2_image_exporter_stac",
            "out_dir": out_dir,
            "version_name": version_name,
        }
        with open(os.path.join(out_dir, f"metadata_{version_name}.json"), "w") as f:
            json.dump(run_meta, f, indent=2)

    data = pd.read_csv(csv)
    if subset:
        data = data.head(10)

    rows = [r for _, r in data.iterrows()]
    total = len(rows)
    logging.info(f"Parallel run: {total} sites, {max_concurrent_sites} concurrent")

    def process_one(row):
        lat, lon = row["y"], row["x"]
        date = datetime(int(row["year"]), int(row["month"]), int(row["day"]))
        date_str = f"{date.year}.{date.month:02d}.{date.day:02d}"
        sid = str(row["site_id"]).replace("id_", "")
        file_id = f"{sid}_{date_str}"

        stack_path = os.path.join(out_dir, f"{file_id}_stack.tif")
        if os.path.exists(stack_path):
            return ("skip", file_id, None)

        try:
            retrieve_time_series_stack(
                file_id=file_id, lat=lat, lon=lon, date=date,
                out_dir=out_dir, collection="L2A",
                start_month=start_month, num_windows=num_windows,
                timestep=timestep, window_buffer=window_buffer,
                target_size=target_size,
                scene_exporter=s2_image_exporter_stac,
            )
            return ("done", file_id, None)
        except Exception as e:
            return ("fail", file_id, str(e))

    done = skipped = failed = 0
    with ThreadPoolExecutor(max_workers=max_concurrent_sites) as ex:
        futures = {ex.submit(process_one, r): r for r in rows}
        for fut in as_completed(futures):
            status, file_id, err = fut.result()
            finished = done + skipped + failed + 1
            if status == "done":
                done += 1
                logging.info(f"[{finished}/{total}] Completed {file_id}")
            elif status == "skip":
                skipped += 1
                logging.info(f"[{finished}/{total}] Skipped (exists) {file_id}")
            else:
                failed += 1
                logging.error(f"[{finished}/{total}] FAILED {file_id}: {err}")

    get_stats(out_dir)
    logging.info(f"Done: {done} completed, {skipped} skipped, {failed} failed")
    return out_dir


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    LABEL_CSV = os.path.join(
        project_root,
        "data/labels/labeled_surveys/random_sample/latest_irrigation_table.csv",
    )
    data_root = get_data_root()
    DOWNLOAD_DIR = os.path.join(data_root, "features/sentinel2")

    dataset_download_parallel(
        csv=LABEL_CSV,
        download_dir=DOWNLOAD_DIR,
        max_concurrent_sites=6,
        subset=False,          # Full ~2,350-site run
        start_month=1,
        num_windows=36,
        timestep=10,
        window_buffer=3,
        target_size=100,
    )
