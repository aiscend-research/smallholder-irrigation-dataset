#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spatially aware data splitting (siteNumeric grouping) with optional stratification.

This script:
1. Do splits at the **site** level first (grouping by siteNumeric), then expand to file stems.
2. Optional stratification by labels from CSV or from *_label.tif Band-2 (irrigation presence).
3. Expose convenience functions that export lists (no moving files) under
  data_root/organized/splits/{split_lists, cv_lists}.
"""

import os
import re
import json
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, StratifiedShuffleSplit
try:
    from sklearn.model_selection import StratifiedGroupKFold
    _HAS_SGF = True
except Exception:
    _HAS_SGF = False

try:
    import rasterio
except Exception:
    rasterio = None

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def _json_safe(obj):
    """Convert numpy scalars and keys to Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(x) for x in obj]
    if hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    return obj


class IrrigationDataSplitter:
    """Split by siteNumeric to avoid spatial leakage; expand sites to file stems."""

    _STEM_RE = re.compile(
        r'^(?P<uid>\d+)_(?P<site>\d+)_(?P<date>\d{4}\.\d{2}\.\d{2})_(?P<kind>image|label)$'
    )

    # Additional regexes for GRIT raw file naming (images/features and masks/labels)
    _GRIT_IMG_RE = re.compile(
        r"^site_[^_]+_[^_]+_\d{4}_(?P<uid>\d+)\.(?P<ext>tif|json)$",
        re.IGNORECASE,
    )
    _GRIT_MASK_RE = re.compile(
        r"^(?P<uid>\d+?)_(?P<site>\d+)_(?P<date>\d{4}\.\d{2}\.\d{2})_(?P<tag>[A-Za-z]+)(_metadata)?\.(?P<ext>tif|json)$",
        re.IGNORECASE,
    )

    def __init__(self, data_root: str, csv_path: Optional[str] = None, random_state: int = 42,
                 grit_images_dir: Optional[str] = None, grit_masks_dir: Optional[str] = None):
        """
        Args:
            data_root: path to "data/modeling" (the folder that contains 'organized').
            csv_path: optional CSV for labels; expected columns: site_id (e.g., id_5168346), irrigation (0/1).
        """
        self.data_root = Path(data_root)
        self.csv_path = csv_path
        self.random_state = random_state

        # Local organized structure (kept for backward compatibility)
        self.organized = self.data_root / "organized"
        self.images_dir = self.organized / "images"
        self.labels_dir = self.organized / "labels"
        self.metadata_dir = self.organized / "metadata"

        # GRIT mode: if both directories are provided, we scan cloud folders instead of local organized/*
        self._grit_images_dir = Path(grit_images_dir).resolve() if grit_images_dir else None
        self._grit_masks_dir = Path(grit_masks_dir).resolve() if grit_masks_dir else None
        self._grit_mode = (self._grit_images_dir is not None) and (self._grit_masks_dir is not None)

        self.df = None
        if csv_path and os.path.exists(csv_path):
            try:
                self.df = pd.read_csv(csv_path)
            except Exception as e:
                logger.warning(f"Failed to read CSV ({csv_path}): {e}")

        self._sites: List[str] = []
        self._site_to_files: Dict[str, List[str]] = {}
        self._y_by_site: Optional[Dict[str, int]] = None

        # In GRIT mode we build a mapping from standardized stem -> absolute cloud paths
        # { stem: {"image_path": Path, "label_path": Path, "json_path": Path} }
        self._stem_to_paths: Dict[str, Dict[str, Path]] = {}

    def _parse_stem(self, stem: str) -> Optional[Dict[str, str]]:
        m = self._STEM_RE.match(stem)
        if not m:
            return None
        gd = m.groupdict()
        try:
            int(gd["uid"])
        except Exception:
            return None
        return gd

    def _scan(self):
        """Scan either GRIT folders or local organized/{images,labels,metadata}; index files by siteNumeric."""
        # GRIT branch: pair feature (image) files and mask (label) files by UID; standardize stems
        if self._grit_mode:
            imgs_by_uid: Dict[str, Dict[str, Path]] = {}
            masks_by_uid: Dict[str, Dict[str, Optional[Path]]] = {}

            # Scan features/images
            if not self._grit_images_dir.exists():
                logger.warning(f"GRIT images dir not found: {self._grit_images_dir}")
            else:
                for fp in self._grit_images_dir.iterdir():
                    if not fp.is_file():
                        continue
                    m = self._GRIT_IMG_RE.match(fp.name)
                    if not m:
                        continue
                    uid = m.group("uid")
                    ext = m.group("ext").lower()
                    rec = imgs_by_uid.setdefault(uid, {"tif": None, "json": None})
                    if ext == "tif":
                        rec["tif"] = fp
                    elif ext == "json":
                        rec["json"] = fp

            # Scan masks/labels
            if not self._grit_masks_dir.exists():
                logger.warning(f"GRIT masks dir not found: {self._grit_masks_dir}")
            else:
                for fp in self._grit_masks_dir.iterdir():
                    if not fp.is_file():
                        continue
                    m = self._GRIT_MASK_RE.match(fp.name)
                    if not m:
                        continue
                    uid = m.group("uid")
                    site = m.group("site")
                    date = m.group("date")
                    ext = m.group("ext").lower()
                    rec = masks_by_uid.setdefault(uid, {"site": site, "date": date, "tif": None, "json": None})
                    if ext == "tif":
                        rec["tif"] = fp
                    elif ext == "json":
                        rec["json"] = fp

            # Pair image/mask by UID and build standardized stems: {uid}_{site}_{YYYY.MM.DD}_image
            by_site: Dict[str, List[str]] = defaultdict(list)
            self._stem_to_paths.clear()

            for uid, img_rec in imgs_by_uid.items():
                mrec = masks_by_uid.get(uid)
                if not mrec:
                    continue
                # --- MOD: make JSON optional; require only both TIFs ---
                if not img_rec.get("tif") or not mrec.get("tif"):
                    continue
                json_path = img_rec.get("json") or mrec.get("json")  # may be None

                site = str(mrec["site"])
                date = str(mrec["date"])
                stem_image = f"{uid}_{site}_{date}_image"

                by_site[site].append(stem_image)
                self._stem_to_paths[stem_image] = {
                    "image_path": img_rec["tif"],
                    # Use mask tif as label_path; this aligns with downstream expectations
                    "label_path": mrec["tif"],
                    # Choose the image JSON as metadata path if present; otherwise mask JSON; may be None
                    "json_path": json_path,
                }

            self._site_to_files = {k: v for k, v in by_site.items()}
            self._sites = sorted(self._site_to_files.keys())

            if not self._sites:
                logger.warning("No paired (image, mask) found under GRIT directories")

            # Optional summary to help debug usable count
            try:
                total_pairs = sum(len(v) for v in self._site_to_files.values())
                logger.info(f"[scan] GRIT sites={len(self._sites)} paired_files={total_pairs}")
            except Exception:
                pass

            return  # End GRIT branch

        # Local organized structure branch (unchanged)
        imgs = list(self.images_dir.glob("*_image.tif"))
        lbls = list(self.labels_dir.glob("*_label.tif"))
        jsons = list(self.metadata_dir.glob("*.json"))

        have_img = {p.stem: p for p in imgs}
        have_lbl = {p.stem: p for p in lbls}
        have_json = {p.stem: p for p in jsons}

        by_site: Dict[str, List[str]] = defaultdict(list)
        for stem in have_img.keys():
            parts = self._parse_stem(stem)
            if not parts:
                continue
            site = parts["site"]
            by_site[site].append(stem)

            # Simple consistency check
            if stem.replace("_image", "_label") not in have_lbl:
                logger.warning(f"Missing label tif for: {stem}")

            if stem not in have_json and stem.replace("_image", "_label") not in have_json:
                logger.warning(f"Missing json for: {stem} (image/label json)")

        self._site_to_files = dict(by_site)
        self._sites = sorted(by_site.keys())
        if not self._sites:
            logger.warning("No *_image.tif found under organized/images")

    def _infer_site_labels(self, mode: str = "csv_then_label", csv_label_col: str = "irrigation"):
        """
        Decide each site's label for stratification.
        mode:
          - 'group_only': no labels (do group-only splitting)
          - 'csv_then_label': prefer CSV's 'irrigation' col; else fallback to label band-2 any>0
          - 'label_band2_any': only use label band-2 any>0
        """
        if mode == "group_only":
            self._y_by_site = None
            return

        y_csv = {}
        if mode.startswith("csv") and self.df is not None and (csv_label_col in self.df.columns):
            for _, r in self.df[["site_id", csv_label_col]].dropna().iterrows():
                sid = str(r["site_id"]).replace("id_", "")
                try:
                    y_csv[sid] = int(r[csv_label_col])
                except Exception:
                    pass

        y: Dict[str, int] = {}
        for site in self._sites:
            if site in y_csv:
                y[site] = y_csv[site]
                continue

            if mode in ("csv_then_label", "label_band2_any"):
                if rasterio is None:
                    y[site] = 0
                    continue

                # In GRIT mode, read label_path from self._stem_to_paths.
                # Otherwise, read from local organized/labels.
                for stem in self._site_to_files.get(site, []):
                    try:
                        if self._grit_mode:
                            rec = self._stem_to_paths.get(stem)
                            if not rec or not rec.get("label_path") or not Path(rec["label_path"]).exists():
                                continue
                            with rasterio.open(rec["label_path"]) as ds:
                                band2 = ds.read(2)
                                if (band2 > 0).any():
                                    y[site] = 1
                                    break
                        else:
                            lbl_path = (self.labels_dir / f"{stem.replace('_image', '_label')}.tif")
                            if not lbl_path.exists():
                                continue
                            with rasterio.open(lbl_path) as ds:
                                band2 = ds.read(2)
                                if (band2 > 0).any():
                                    y[site] = 1
                                    break
                    except Exception:
                        # On any read error, continue scanning other stems of the same site
                        continue

                # Default to 0 if nothing set
                if site not in y:
                    y[site] = 0
            else:
                y[site] = 0
        self._y_by_site = y

    # lists-only exporters
    def export_split_lists(self,
                           output_dir: str,
                           test_size: float = 0.2,
                           val_size: float = 0.2,
                           y_mode: str = "csv_then_label",
                           min_samples_per_class: int = 5) -> Dict:
        """
        One-shot train/val/test split at the site level with optional stratification.
        Writes text lists and a manifest.csv to output_dir. No file movement.
        """
        self._scan()
        self._infer_site_labels(mode=y_mode)

        sites = np.array(self._sites)
        if len(sites) < 3:
            all_files = [f for s in sites for f in self._site_to_files.get(s, [])]
            meta = {
                "total_sites": int(len(sites)),
                "train_sites": int(len(sites)), "val_sites": 0, "test_sites": 0,
                "total_files": int(len(all_files)), "stratified": False,
                "warning": "Insufficient sites; using all for training"
            }
            return self._write_lists(output_dir, train=all_files, val=[], test=[], metadata=meta)

        rng = np.random.RandomState(self.random_state)
        if (self._y_by_site is None):
            sites_shuf = sites.copy()
            rng.shuffle(sites_shuf)
            n_test = max(1, int(round(test_size * len(sites_shuf))))
            test_sites = sites_shuf[:n_test]
            remain = sites_shuf[n_test:]
            n_val = max(1, int(round(val_size * len(remain)))) if len(remain) > 1 else 0
            val_sites = remain[:n_val]
            train_sites = remain[n_val:]
            stratified = False
        else:
            y = np.array([self._y_by_site[s] for s in sites])
            uniq, cnt = np.unique(y, return_counts=True)
            ok = {c for c, k in zip(uniq, cnt) if k >= min_samples_per_class}
            if len(ok) < 2:
                sites_shuf = sites.copy()
                rng.shuffle(sites_shuf)
                n_test = max(1, int(round(test_size * len(sites_shuf))))
                test_sites = sites_shuf[:n_test]
                remain = sites_shuf[n_test:]
                n_val = max(1, int(round(val_size * len(remain)))) if len(remain) > 1 else 0
                val_sites = remain[:n_val]
                train_sites = remain[n_val:]
                stratified = False
            else:
                mask_ok = np.isin(y, list(ok))
                sites_ok, y_ok = sites[mask_ok], y[mask_ok]
                sss_outer = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=self.random_state)
                tr_idx, te_idx = next(sss_outer.split(sites_ok, y_ok))
                train_pool, test_sites = sites_ok[tr_idx], sites_ok[te_idx]
                y_train_pool = y_ok[tr_idx]
                sss_inner = StratifiedShuffleSplit(n_splits=1, test_size=val_size/(1.0-test_size),
                                                   random_state=self.random_state)
                tr2_idx, va_idx = next(sss_inner.split(train_pool, y_train_pool))
                train_sites, val_sites = train_pool[tr2_idx], train_pool[va_idx]
                stratified = True

        train_files = [f for s in train_sites for f in self._site_to_files.get(s, [])]
        val_files   = [f for s in val_sites   for f in self._site_to_files.get(s, [])]
        test_files  = [f for s in test_sites  for f in self._site_to_files.get(s, [])]

        assert set(train_files).isdisjoint(val_files)
        assert set(train_files).isdisjoint(test_files)
        assert set(val_files).isdisjoint(test_files)

        meta = {
            "total_sites": int(len(self._sites)),
            "train_sites": int(len(train_sites)),
            "val_sites": int(len(val_sites)),
            "test_sites": int(len(test_sites)),
            "total_files": int(len(train_files) + len(val_files) + len(test_files)),
            "stratified": stratified,
            "y_mode": y_mode
        }
        if stratified and (self._y_by_site is not None):
            def dist(sites_sel):
                arr = np.array([self._y_by_site[s] for s in sites_sel])
                return {str(k): int(v) for k, v in zip(*np.unique(arr, return_counts=True))}
            meta["class_distribution"] = {
                "train": dist(train_sites), "val": dist(val_sites), "test": dist(test_sites)
            }

        return self._write_lists(output_dir, train=train_files, val=val_files, test=test_files, metadata=meta)

    def export_cv_lists_with_heldout_test(self,
                                          output_dir: str,
                                          n_splits: int = 5,
                                          test_size: float = 0.2,
                                          y_mode: str = "csv_then_label",
                                          min_samples_per_class: int = 5) -> Dict:
        """
        Held-out test + K-fold CV on the training pool. Writes lists only.
        """
        one = self.export_split_lists(
            output_dir=output_dir, test_size=test_size, val_size=0.0,
            y_mode=y_mode, min_samples_per_class=min_samples_per_class
        )
        test_files = one["test_files"]
        train_pool_files = one["train_files"]

        # Build site list for train pool
        def site_from_stem(stem: str) -> Optional[str]:
            m = self._STEM_RE.match(stem)
            return m.group("site") if m else None

        file_to_site = {stem: site_from_stem(stem) for stem in train_pool_files}
        train_pool_sites = sorted({s for s in file_to_site.values() if s is not None})

        # CV at site level
        self._scan()
        self._infer_site_labels(mode=y_mode)
        sites_all = np.array(self._sites)
        sites_cv = np.array([s for s in sites_all if s in set(train_pool_sites)])
        y = None if self._y_by_site is None else np.array([self._y_by_site[s] for s in sites_cv])
        groups = sites_cv

        if (y is None) or (len(np.unique(y)) < 2) or (not _HAS_SGF):
            splitter = GroupKFold(n_splits=n_splits)
            split_iter = splitter.split(X=sites_cv, y=None, groups=groups)
            stratified = False
        else:
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            split_iter = splitter.split(X=sites_cv, y=y, groups=groups)
            stratified = True

        # Write fold lists
        root = Path(output_dir)
        train_root = root / "train"
        train_root.mkdir(parents=True, exist_ok=True)

        cv_meta = []
        used = set(test_files)
        for i, (tr_idx, va_idx) in enumerate(split_iter, start=1):
            tr_sites = sites_cv[tr_idx].tolist()
            va_sites = sites_cv[va_idx].tolist()
            tr_files = [f for s in tr_sites for f in self._site_to_files.get(s, []) if f in train_pool_files]
            va_files = [f for s in va_sites for f in self._site_to_files.get(s, []) if f in train_pool_files]

            assert set(tr_files).isdisjoint(va_files)

            fold_dir = train_root / f"fold_{i}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            (fold_dir / "train_files.txt").write_text("\n".join(tr_files), encoding="utf-8")
            (fold_dir / "val_files.txt").write_text("\n".join(va_files), encoding="utf-8")

            used.update(tr_files); used.update(va_files)
            meta = {
                "fold": i, "train_sites": int(len(tr_sites)), "val_sites": int(len(va_sites)),
                "train_files": int(len(tr_files)), "val_files": int(len(va_files)), "stratified": stratified
            }
            if stratified and (y is not None):
                meta["class_distribution"] = {
                    "train": {str(k): int(v) for k, v in zip(*np.unique(y[tr_idx], return_counts=True))},
                    "val":   {str(k): int(v) for k, v in zip(*np.unique(y[va_idx], return_counts=True))},
                }
            cv_meta.append(meta)

        # Save manifest.csv for all used stems (paths are cloud absolute paths in GRIT mode)
        used_list = list(used)
        manifest = self._make_manifest_df(used_list)
        try:
            manifest.to_csv(root / "manifest.csv", index=False)
        except Exception as e:
            logger.warning(f"Failed to write manifest.csv: {e}")

        # Save test list and metadata
        (root / "test_files.txt").write_text("\n".join(test_files), encoding="utf-8")
        overall = {"n_splits": n_splits, "y_mode": y_mode, "heldout_test_files": int(len(test_files)), "cv_folds": cv_meta}
        with open(root / "cv_metadata.json", "w", encoding="utf-8") as f:
            json.dump(_json_safe(overall), f, indent=2)

        return {"root": str(root), "test_files": test_files, "cv_meta": cv_meta}

    # helpers
    def _make_manifest_df(self, stems: List[str]) -> "pd.DataFrame":
        rows = []
        for stem in stems:
            m = self._STEM_RE.match(stem)
            if not m:
                continue
            gd = m.groupdict()

            if self._grit_mode and stem in self._stem_to_paths:
                rec = self._stem_to_paths[stem]
                # --- MOD: handle optional json_path safely (might be None) ---
                jp = rec.get("json_path")
                jp_str = str(Path(jp).resolve()) if jp else ""
                rows.append({
                    "stem": stem,
                    "image_path": str(Path(rec["image_path"]).resolve()),
                    "label_path": str(Path(rec["label_path"]).resolve()),
                    "json_path":  jp_str,
                    "site": gd["site"], "uid": gd["uid"], "date": gd["date"]
                })
            else:
                rows.append({
                    "stem": stem,
                    "image_path": str((self.images_dir / f"{stem}.tif").resolve()),
                    "label_path": str((self.labels_dir / f"{stem.replace('_image', '_label')}.tif").resolve()),
                    "json_path":  str((self.metadata_dir / f"{stem}.json").resolve()),
                    "site": gd["site"], "uid": gd["uid"], "date": gd["date"]
                })
        return pd.DataFrame(rows)

    def _write_lists(self, output_dir: str, train: List[str], val: List[str], test: List[str], metadata: Dict) -> Dict:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        (root / "train_files.txt").write_text("\n".join(train), encoding="utf-8")
        (root / "val_files.txt").write_text("\n".join(val), encoding="utf-8")
        (root / "test_files.txt").write_text("\n".join(test), encoding="utf-8")

        used = list(dict.fromkeys(train + val + test))
        manifest = self._make_manifest_df(used)
        try:
            manifest.to_csv(root / "manifest.csv", index=False)
        except Exception as e:
            logger.warning(f"Failed to write manifest.csv: {e}")

        with open(root / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(_json_safe(metadata), f, indent=2)

        return {"root": str(root), "train_files": train, "val_files": val, "test_files": test, "metadata": metadata}


def prepare_and_export_splits(data_root: str,
                              csv_path: Optional[str] = None,
                              y_mode: str = "csv_then_label",
                              n_splits: int = 5,
                              test_size: float = 0.2,
                              val_size: float = 0.2,
                              min_samples_per_class: int = 5,
                              grit_images_dir: Optional[str] = None,
                              grit_masks_dir: Optional[str] = None) -> Dict:
    """
    Export both one-shot split and CV lists under data_root/organized/splits.
    Returns a dict with useful paths for downstream training.
    """
    data_root = Path(data_root).resolve()
    splits_root = data_root / "organized" / "splits"
    one_shot_dir = splits_root / "split_lists"
    cv_dir = splits_root / "cv_lists"
    one_shot_dir.mkdir(parents=True, exist_ok=True)
    cv_dir.mkdir(parents=True, exist_ok=True)

    splitter = IrrigationDataSplitter(str(data_root), csv_path=csv_path,
                                      grit_images_dir=grit_images_dir,
                                      grit_masks_dir=grit_masks_dir)
    # one-shot
    one = splitter.export_split_lists(
        output_dir=str(one_shot_dir),
        test_size=test_size, val_size=val_size,
        y_mode=y_mode, min_samples_per_class=min_samples_per_class
    )
    # cv
    cv = splitter.export_cv_lists_with_heldout_test(
        output_dir=str(cv_dir),
        n_splits=n_splits, test_size=test_size,
        y_mode=y_mode, min_samples_per_class=min_samples_per_class
    )

    return {
        "one_shot_dir": str(one_shot_dir),
        "cv_dir": str(cv_dir),
        "train_list": str(one_shot_dir / "train_files.txt"),
        "val_list": str(one_shot_dir / "val_files.txt"),
        "test_list": str(one_shot_dir / "test_files.txt"),
        "manifest_csv": str(one_shot_dir / "manifest.csv"),
        "cv_test_list": str(cv_dir / "test_files.txt"),
        "cv_manifest_csv": str(cv_dir / "manifest.csv"),
    }