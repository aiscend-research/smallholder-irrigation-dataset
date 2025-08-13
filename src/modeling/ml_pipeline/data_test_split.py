import argparse
import os
import re
import json
import shutil
from typing import Dict, List, Tuple, Optional
from pathlib import Path

IMAGES_DIR = "/home/waves/data/smallholder-irrigation-dataset/data/features/"
MASKS_DIR = "/home/waves/data/smallholder-irrigation-dataset/data/masks/labels"
OUT_ROOT = "/home/waves/data/smallholder-irrigation-dataset/data/modeling"
VAL_FRAC = 0.2
MAX_SAMPLES = 50

def _scan_images(images_dir: str) -> List[Dict]:
    pattern = re.compile(
        r"^site_[^_]+_[^_]+_\d{4}_(?P<uid>\d+)\.(?P<ext>tif|json)$",
        re.IGNORECASE,
    )
    files_by_uid = {}
    images_path = Path(images_dir)
    for file_path in images_path.iterdir():
        if not file_path.is_file():
            continue
        print(f"Raw image file: {file_path.name}")
        m = pattern.match(file_path.name)
        if not m:
            print(f"No match for image file: {file_path.name}")
            continue
        uid = m.group("uid")
        ext = m.group("ext").lower()
        print(f"Image match: {file_path.name} -> UID {uid}")
        if uid not in files_by_uid:
            files_by_uid[uid] = {"uid": uid, "tif": None, "json": None}
        if ext == "tif":
            files_by_uid[uid]["tif"] = file_path
        elif ext == "json":
            files_by_uid[uid]["json"] = file_path
    return list(files_by_uid.values())

def _scan_masks(masks_dir: str) -> List[Dict]:
    pattern = re.compile(
        r"^(?P<uid>\d+?)_(?P<site_id>\d+)_(?P<date>\d{4}\.\d{2}\.\d{2})_(?P<tag>[A-Za-z]+)\.(?P<ext>tif|json)$",
        re.IGNORECASE,
    )
    masks_by_uid = {}
    masks_path = Path(masks_dir)
    for file_path in masks_path.iterdir():
        if not file_path.is_file():
            continue
        m = pattern.match(file_path.name)
        if not m:
            continue
        uid = m.group("uid")
        site_id = m.group("site_id")
        date = m.group("date")
        ext = m.group("ext").lower()
        if uid not in masks_by_uid:
            masks_by_uid[uid] = {"uid": uid, "site_id": site_id, "date": date, "tif": None, "json": None}
        if ext == "tif":
            masks_by_uid[uid]["tif"] = file_path
        elif ext == "json":
            masks_by_uid[uid]["json"] = file_path
    # Filter out entries missing tif or json
    result = []
    for mask in masks_by_uid.values():
        if mask["tif"] and mask["json"]:
            result.append(mask)
    return result

def _pair_records(images: List[Dict], masks: List[Dict]) -> List[Dict]:
    pairs = []
    masks_by_uid = {m['uid']: m for m in masks}
    for img in images:
        uid = img.get('uid')
        
        #DEBUG
        print(f"Trying to pair UID {uid}: "
              f"Image tif? {bool(img.get('tif'))},"
              f"image json? {bool(img.get('json'))}, "
              f"mask exists? {uid in masks_by_uid}")
        
        mask_best = masks_by_uid.get(uid)
        if not img or not img.get('tif') or not img.get('json'):
            continue  # require image tif AND json
        if not mask_best or not mask_best.get('tif') or not mask_best.get('json'):
            continue  # require mask tif AND json
        # Use mask-provided site_id and full date only; no fallbacks
        site_id = mask_best['site_id']
        date = mask_best['date']
        pairs.append({
            'uid': uid,
            'site_id': site_id,
            'date': date,
            'image_tif': img['tif'],
            'image_json': img['json'],
            'mask_tif': mask_best['tif'],
            'mask_json': mask_best['json']
        })
        print(f"Paired UID {uid}: image=({img['tif'].name}, {img['json'].name}), mask=({mask_best['tif'].name}, {mask_best['json'].name})")
    return pairs

def _copy_pair(pair: Dict, dst_dir: str):
    base = pair['uid']
    # image tif
    dst_img_tif = dst_dir / f"{base}_image.tif"
    shutil.copy2(str(pair['image_tif']), str(dst_img_tif))
    # image json
    dst_img_json = dst_dir / f"{base}_image.json"
    shutil.copy2(str(pair['image_json']), str(dst_img_json))
    # mask tif
    dst_msk_tif = dst_dir / f"{base}_mask.tif"
    shutil.copy2(str(pair['mask_tif']), str(dst_msk_tif))
    # mask json
    dst_msk_json = dst_dir / f"{base}_mask.json"
    shutil.copy2(str(pair['mask_json']), str(dst_msk_json))

def main():
    images = _scan_images(IMAGES_DIR)
    masks = _scan_masks(MASKS_DIR)
    pairs = _pair_records(images, masks)

    # Limit pairs to MAX_SAMPLES
    pairs = pairs[:MAX_SAMPLES]

    # Split into train and val
    val_count = int(len(pairs) * VAL_FRAC)
    train_pairs = pairs[val_count:]
    val_pairs = pairs[:val_count]

    from pathlib import Path
    train_dir = Path(OUT_ROOT) / "train"
    val_dir = Path(OUT_ROOT) / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    for pair in train_pairs:
        _copy_pair(pair, train_dir)
    for pair in val_pairs:
        _copy_pair(pair, val_dir)

    print(f"Total pairs: {len(pairs)}")
    print(f"Train count: {len(train_pairs)}")
    print(f"Val count: {len(val_pairs)}")

if __name__ == "__main__":
    main()
