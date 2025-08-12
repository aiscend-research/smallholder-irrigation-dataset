import argparse
import os
import re
import json
import shutil
from typing import Dict, List, Tuple, Optional

IMAGES_DIR = "/home/waves/data/smallholder-irrigation-dataset/data/features/"
MASKS_DIR = "/home/waves/data/smallholder-irrigation-dataset/data/masks/labels"
OUT_ROOT = "/home/waves/data/smallholder-irrigation-dataset/data/modeling"
VAL_FRAC = 0.2
MAX_SAMPLES = 50
COPY_MODE = "copy"  # or "link", "symlink"
DRY_RUN = False


def _copy(src: str, dst: str, mode: str):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "link":
        if os.path.exists(dst):
            os.remove(dst)
        os.link(src, dst)
    elif mode == "symlink":
        if os.path.exists(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(src), dst)
    else:
        raise ValueError(f"Unknown copy mode: {mode}")

def _gather_images(images_dir: str) -> Dict[int, Tuple[str, str]]:
    """
    Returns mapping: unique_id -> (image_tif_path, image_json_path)
    We expect a .json and a .tif sharing the same stem (minus extension).
    Unique id is the first number before the first underscore in the filename.
    """
    mapping: Dict[int, Tuple[str, str]] = {}
    for root, _, files in os.walk(images_dir):
        for f in files:
            if not f.lower().endswith(".json"):
                continue
            base = os.path.basename(f)
            m = re.match(r"^(\d+)_", base)
            if not m:
                continue
            unique_id = int(m.group(1))
            stem = f[:-5]  # drop .json
            tif_guess = os.path.join(root, stem + ".tif")
            if not os.path.exists(tif_guess):
                # try .tiff
                tif_guess2 = os.path.join(root, stem + ".tiff")
                if os.path.exists(tif_guess2):
                    tif_guess = tif_guess2
                else:
                    # no image stack found
                    continue
            mapping[unique_id] = (tif_guess, os.path.join(root, f))
    return mapping

def _parse_mask_name(mask_tif_name: str) -> Tuple[str, str, str]:
    """
    Parse mask filename of the form:
      '<unique>_<site>_<YYYY.MM.DD>[ _<Suffix> ].tif(f)'
    The trailing analyst/code suffix after the date is optional and may be any
    non-space string without dots (e.g., KL, MV, JL, DSB, etc.).

    Examples:
      '1_5168346_2023.09.06_KL.tif'      -> ('1','5168346','2023.09.06')
      '2_5168346_2019.10.30.tif'         -> ('2','5168346','2019.10.30')
      '/path/3_3581818_2024.06.30_JL.tiff' -> ('3','3581818','2024.06.30')
    """
    base = os.path.basename(mask_tif_name)
    m = re.match(r"^(\d+)_(\d+)_(\d{4}\.\d{2}\.\d{2})(?:_[^.\s]+)?\.(?:tif|tiff)$", base, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Mask name not in expected form '<unique>_<site>_<YYYY.MM.DD>[_SUFFIX].tif(f)': {mask_tif_name}")
    unique_id, site_id, date_str = m.group(1), m.group(2), m.group(3)
    return unique_id, site_id, date_str

def _find_mask_metadata(mask_tif_path: str) -> Optional[str]:
    """
    For '.../1_5168346_2023.09.06_KL.tif' expects sibling
    '.../1_5168346_2023.09.06_KL_metadata.json'
    """
    stem = os.path.splitext(mask_tif_path)[0]
    meta_path = f"{stem}_metadata.json"
    return meta_path if os.path.exists(meta_path) else None

def _train_val_split(indices: List[int], val_frac: float) -> Tuple[List[int], List[int]]:
    n = len(indices)
    n_val = int(round(n * val_frac))
    val_idx = indices[-n_val:] if n_val > 0 else []
    train_idx = indices[: n - n_val]
    return train_idx, val_idx

def main():
    images_map = _gather_images(IMAGES_DIR)
    if not images_map:
        raise SystemExit(f"No image json+tif pairs discovered under: {IMAGES_DIR}")

    # Index all masks
    mask_records = []
    for root, _, files in os.walk(MASKS_DIR):
        for f in files:
            if not f.lower().endswith(".tif"):
                continue
            mask_path = os.path.join(root, f)
            try:
                unique_id, site_id, date_str = _parse_mask_name(mask_path)
            except ValueError as e:
                print(f"[WARN] {e}")
                continue
            mask_records.append({
                "unique": unique_id,
                "site": site_id,
                "date": date_str,             # 'YYYY.MM.DD'
                "mask_tif": mask_path,
            })

    if not mask_records:
        raise SystemExit(f"No masks discovered under: {MASKS_DIR}")

    # Sort by numeric unique id (derived from filename) so selection is deterministic
    mask_records.sort(key=lambda r: int(r["unique"]))

    # Pair with images via unique id from mask record -> images_map
    paired = []
    for rec in mask_records:
        try:
            unique = int(rec["unique"])
        except Exception:
            print(f"[WARN] Could not parse unique id for mask: {rec}")
            continue
        if unique not in images_map:
            print(f"[WARN] Could not pair mask (unique={rec['unique']}) — unique id not found in images.")
            continue
        img_tif, img_json = images_map[unique]
        paired.append((rec, img_tif, img_json))

    if not paired:
        raise SystemExit("No pairs could be made. Ensure mask filenames unique ids exist in image json names.")

    # Sort paired by mask record's numeric unique id for deterministic naming
    paired.sort(key=lambda tup: int(tup[0]["unique"]))

    # Keep only first N if requested (after pairing and sorting)
    if MAX_SAMPLES is not None:
        paired = paired[:MAX_SAMPLES]

    # Split 80/20 (or val_frac), preserving order; unique_id stays from mask name
    train_idx, val_idx = _train_val_split(list(range(len(paired))), VAL_FRAC)

    def emit(split_name: str, items: List[Tuple[dict, str, str]]):
        out_split_dir = os.path.join(OUT_ROOT, split_name)
        os.makedirs(out_split_dir, exist_ok=True)

        for rec, img_tif, img_json in items:
            prefix = f"{rec['unique']}_{rec['site']}_{rec['date']}"  # e.g., '1_5168346_2023.09.06'

            dst_img  = os.path.join(out_split_dir, f"{prefix}_image.tif")
            dst_lab  = os.path.join(out_split_dir, f"{prefix}_label.tif")
            dst_img_json = os.path.join(out_split_dir, f"{prefix}_image.json")
            dst_lab_json = os.path.join(out_split_dir, f"{prefix}_label.json")

            if DRY_RUN:
                print(f"[DRY] {img_tif}  -> {dst_img}")
                print(f"[DRY] {rec['mask_tif']} -> {dst_lab}")
                print(f"[DRY] {img_json} -> {dst_img_json} (image json)")
                meta_path = _find_mask_metadata(rec["mask_tif"])
                if meta_path is not None:
                    print(f"[DRY] {meta_path} -> {dst_lab_json} (mask json)")
            else:
                _copy(img_tif, dst_img, COPY_MODE)
                _copy(rec["mask_tif"], dst_lab, COPY_MODE)
                _copy(img_json, dst_img_json, COPY_MODE)
                meta_path = _find_mask_metadata(rec["mask_tif"])
                if meta_path is not None:
                    _copy(meta_path, dst_lab_json, COPY_MODE)

    emit("train", [paired[i] for i in train_idx])
    emit("val",   [paired[i] for i in val_idx])

    print(f"Done. Wrote dataset to: {OUT_ROOT}")
    print(f" Train: {len(train_idx)} samples  |  Val: {len(val_idx)} samples")


if __name__ == "__main__":
    main()