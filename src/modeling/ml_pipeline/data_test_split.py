import argparse
import os
import shutil
from pathlib import Path


def create_dirs(base_dir):
    for sub_dir in ['images', 'labels', 'json']:
        dir_path = base_dir / sub_dir
        dir_path.mkdir(parents=True, exist_ok=True)


def copy_files(files, src_dir, dst_dir, mode):
    for file in files:
        src_file = src_dir / file
        if file.endswith('.jpg') or file.endswith('.png'):
            subfolder = 'images'
        elif file.endswith('.txt'):
            subfolder = 'labels'
        elif file.endswith('.json'):
            subfolder = 'json'
        else:
            continue

        dst_file = dst_dir / subfolder / file
        if mode == 'copy':
            shutil.copy2(src_file, dst_file)
        elif mode == 'link':
            os.link(src_file, dst_file)
        elif mode == 'symlink':
            os.symlink(src_file, dst_file)


def main(args):
    src_dir = Path(args.source)
    train_files = args.train_files
    val_files = args.val_files

    train_dir = Path(args.output) / 'train_dataset'
    val_dir = Path(args.output) / 'val_dataset'

    if not args.dry_run:
        create_dirs(train_dir)
        create_dirs(val_dir)

        copy_files(train_files, src_dir, train_dir, args.mode)
        copy_files(val_files, src_dir, val_dir, args.mode)
    else:
        print(f'Dry run mode: would create directories {train_dir} and {val_dir} with subfolders images, labels, json')
        print(f'Would copy/link/symlink train files to {train_dir}')
        print(f'Would copy/link/symlink val files to {val_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split dataset into train and val with structured directories.')
    parser.add_argument('--source', type=str, required=True, help='Source directory containing files.')
    parser.add_argument('--output', type=str, required=True, help='Output directory for split dataset.')
    parser.add_argument('--train-files', nargs='+', required=True, help='List of train files.')
    parser.add_argument('--val-files', nargs='+', required=True, help='List of val files.')
    parser.add_argument('--mode', choices=['copy', 'link', 'symlink'], default='copy', help='File operation mode.')
    parser.add_argument('--dry-run', action='store_true', help='Do not perform any file operations.')
    args = parser.parse_args()

    main(args)

#!/usr/bin/env python3
"""
data_test_split.py

Utility to:
1) Normalize "bad" raw images/masks into the modeling convention:
      <PREFIX>_image.tif, <PREFIX>_label.tif, <PREFIX>.json
   where PREFIX = "<unique_id>_<site_id>_<YYYY.MM.DD>"

2) Pair images and masks using the numeric `internal_id` parsed from the image
   metadata json name (e.g., 'site_-10.00_28.75_2019_1703.json' -> internal_id=1703)
   matched against the *mask* metadata json content field `internal_id`.

3) Split into train/val (default 80/20) and (optionally) downselect to the first N
   samples while preserving original `unique_id` from the mask filename. We DO NOT
   reassign or shuffle `unique_id` — it is taken from the mask filename prefix.

Expected inputs:
- images_dir: directory of raw image stacks (*.tif) with accompanying metadata jsons
              named like: 'site_<lat>_<lon>_<year>_<internal>.json'
              and matching .tif with the same basename.
- masks_dir : directory of mask rasters named like: '<unique>_<siteId>_<YYYY.MM.DD>_KL.tif'
              and mask metadata jsons named like: '<same>_metadata.json'
Outputs (under out_root):
  out_root/
    train/images/, train/labels/, train/json/
    val/images/,   val/labels/,   val/json/
All files are *copied* by default (use --link for hardlinks; --symlink for symlinks).

Usage:
  python -m modeling.ml_pipeline.data_test_split \
      --images-dir /path/to/images \
      --masks-dir  /path/to/masks \
      --out-root   /path/to/paired_dataset \
      --val-frac 0.2 \
      --max-samples 200 \
      --seed 42 \
      [--link | --symlink] \
      [--dry-run]
"""

import argparse
import os
import re
import json
import shutil
from typing import Dict, List, Tuple, Optional

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

def _parse_internal_id_from_image_json_name(name: str) -> Optional[int]:
    """
    From a filename like 'site_-10.00_28.75_2019_1703.json' extract 1703.
    Returns None if no trailing integer found.
    """
    base = os.path.basename(name)
    m = re.search(r"_([0-9]+)\.json$", base)
    return int(m.group(1)) if m else None

def _gather_images(images_dir: str) -> Dict[int, Tuple[str, str]]:
    """
    Returns mapping: internal_id -> (image_tif_path, image_json_path)
    We expect a .json and a .tif sharing the same stem (minus extension).
    """
    mapping: Dict[int, Tuple[str, str]] = {}
    # Index all JSONs first
    for root, _, files in os.walk(images_dir):
        for f in files:
            if not f.lower().endswith(".json"):
                continue
            json_path = os.path.join(root, f)
            internal_id = _parse_internal_id_from_image_json_name(f)
            if internal_id is None:
                continue
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
            mapping[internal_id] = (tif_guess, json_path)
    return mapping

def _parse_mask_name(mask_tif_name: str) -> Tuple[str, str, str]:
    """
    From '1_5168346_2023.09.06_KL.tif' -> (unique='1', site='5168346', date='2023.09.06')
    """
    stem = os.path.basename(mask_tif_name)
    stem = os.path.splitext(stem)[0]
    # Drop optional trailing '_KL'
    if stem.endswith("_KL"):
        stem_core = stem[:-3]
    else:
        stem_core = stem
    parts = stem_core.split("_")
    if len(parts) < 3:
        raise ValueError(f"Mask name not in expected form '<unique>_<site>_<YYYY.MM.DD>[_KL].tif': {mask_tif_name}")
    unique_id, site_id, date_str = parts[0], parts[1], parts[2]
    return unique_id, site_id, date_str

def _find_mask_metadata(mask_tif_path: str) -> Optional[str]:
    """
    For '.../1_5168346_2023.09.06_KL.tif' expects sibling
    '.../1_5168346_2023.09.06_KL_metadata.json'
    """
    stem = os.path.splitext(mask_tif_path)[0]
    meta_path = f"{stem}_metadata.json"
    return meta_path if os.path.exists(meta_path) else None

def _read_internal_id_from_mask_metadata(mask_meta_json: str) -> Optional[int]:
    """
    Try to read 'internal_id' from the mask metadata json content if present.
    """
    try:
        with open(mask_meta_json, "r") as f:
            data = json.load(f)
        # allow different casings/keys
        for key in ("internal_id", "internalId", "id_internal", "image_number", "image_id"):
            if key in data and isinstance(data[key], (int, str)):
                try:
                    return int(str(data[key]))
                except Exception:
                    continue
    except Exception:
        pass
    return None

def _select_indices(uids_sorted: List[str], max_samples: Optional[int]) -> List[int]:
    idxs = list(range(len(uids_sorted)))
    if max_samples is not None:
        idxs = idxs[: max_samples]
    return idxs

def _train_val_split(indices: List[int], val_frac: float) -> Tuple[List[int], List[int]]:
    n = len(indices)
    n_val = int(round(n * val_frac))
    val_idx = indices[-n_val:] if n_val > 0 else []
    train_idx = indices[: n - n_val]
    return train_idx, val_idx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", required=True, help="Folder with image stacks and site_*.json metadata")
    ap.add_argument("--masks-dir", required=True, help="Folder with mask tif and *_metadata.json")
    ap.add_argument("--out-root", required=True, help="Output root folder for paired dataset")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--max-samples", type=int, default=None, help="Keep at most N samples (after ordering by unique_id)")
    ap.add_argument("--seed", type=int, default=42, help="Not used for IDs; only for potential future shuffles")
    mode_group = ap.add_mutually_exclusive_group()
    mode_group.add_argument("--copy", action="store_true", help="Default: copy files")
    mode_group.add_argument("--link", action="store_true", help="Use hard links")
    mode_group.add_argument("--symlink", action="store_true", help="Use symlinks")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    copy_mode = "copy"
    if args.link:
        copy_mode = "link"
    elif args.symlink:
        copy_mode = "symlink"

    images_map = _gather_images(args.images_dir)
    if not images_map:
        raise SystemExit(f"No image json+tif pairs discovered under: {args.images_dir}")

    # Index all masks
    mask_records = []
    for root, _, files in os.walk(args.masks_dir):
        for f in files:
            if not f.lower().endswith(".tif"):
                continue
            if "_KL" not in f:  # safety net; keep the convention
                continue
            mask_path = os.path.join(root, f)
            meta_path = _find_mask_metadata(mask_path)
            if meta_path is None:
                print(f"[WARN] No metadata json for mask: {mask_path}")
                continue
            try:
                unique_id, site_id, date_str = _parse_mask_name(mask_path)
            except ValueError as e:
                print(f"[WARN] {e}")
                continue
            internal_id = _read_internal_id_from_mask_metadata(meta_path)
            mask_records.append({
                "unique": unique_id,
                "site": site_id,
                "date": date_str,             # 'YYYY.MM.DD'
                "mask_tif": mask_path,
                "mask_json": meta_path,
                "internal_id": internal_id,   # may be None
            })

    if not mask_records:
        raise SystemExit(f"No masks discovered under: {args.masks_dir}")

    # Sort by numeric unique id (derived from filename) so selection is deterministic
    mask_records.sort(key=lambda r: int(r["unique"]))

    # Keep only first N if requested
    sel_indices = _select_indices([r["unique"] for r in mask_records], args.max_samples)
    selected = [mask_records[i] for i in sel_indices]

    # Pair with images via internal_id from mask metadata -> images_map
    paired = []
    for rec in selected:
        iid = rec["internal_id"]
        if iid is None or iid not in images_map:
            print(f"[WARN] Could not pair mask (unique={rec['unique']}) — internal_id missing or not found in images.")
            continue
        img_tif, img_json = images_map[iid]
        paired.append((rec, img_tif, img_json))

    if not paired:
        raise SystemExit("No pairs could be made. Ensure mask metadata has an 'internal_id' that exists in image json names.")

    # Split 80/20 (or val_frac), preserving order; unique_id stays from mask name
    train_idx, val_idx = _train_val_split(list(range(len(paired))), args.val_frac)

    def emit(split_name: str, items: List[Tuple[dict, str, str]]):
        out_images = os.path.join(args.out_root, split_name, "images")
        out_labels = os.path.join(args.out_root, split_name, "labels")
        out_json   = os.path.join(args.out_root, split_name, "json")
        os.makedirs(out_images, exist_ok=True)
        os.makedirs(out_labels, exist_ok=True)
        os.makedirs(out_json, exist_ok=True)

        for rec, img_tif, img_json in items:
            prefix = f"{rec['unique']}_{rec['site']}_{rec['date']}"  # e.g., '1_5168346_2023.09.06'

            dst_img  = os.path.join(out_images, f"{prefix}_image.tif")
            dst_lab  = os.path.join(out_labels, f"{prefix}_label.tif")
            dst_json = os.path.join(out_json,   f"{prefix}.json")

            if args.dry_run:
                print(f"[DRY] {img_tif}  -> {dst_img}")
                print(f"[DRY] {rec['mask_tif']} -> {dst_lab}")
                print(f"[DRY] {img_json} -> {dst_json} (image json preferred; will fallback to mask json if missing)")
            else:
                _copy(img_tif, dst_img, copy_mode)
                _copy(rec["mask_tif"], dst_lab, copy_mode)
                # Prefer the image json as the canonical metadata. If missing, use mask json.
                json_src = img_json if os.path.exists(img_json) else rec["mask_json"]
                _copy(json_src, dst_json, copy_mode)

    emit("train", [paired[i] for i in train_idx])
    emit("val",   [paired[i] for i in val_idx])

    print(f"Done. Wrote dataset to: {args.out_root}")
    print(f" Train: {len(train_idx)} samples  |  Val: {len(val_idx)} samples")

if __name__ == "__main__":
    main()