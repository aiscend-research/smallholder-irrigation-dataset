#!/usr/bin/env python3
"""
This module implements spatial-aware data splitting strategies that:
1. Split by location to avoid spatial data leakage
2. Maintain class balance through stratified sampling
3. Support experimentation with different label bands
4. Handle the 8-band irrigation label structure
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ---------------- GRIT paths (override as needed) ----------------
IMAGES_DIR = "/home/waves/data/smallholder-irrigation-dataset/data/features"
MASKS_DIR  = "/home/waves/data/smallholder-irrigation-dataset/data/masks/labels"
OUT_ROOT   = "/home/waves/data/smallholder-irrigation-dataset/data/modeling"
# Example CSV (update as appropriate)
# CSV_PATH = "/home/waves/data/smallholder-irrigation-dataset/data/labels/labeled_surveys/random_sample/latest_irrigation_table.csv"

# ---------------- Part-1 patterns (brought in) ----------------
IMG_RX = re.compile(
    r"^site_[^_]+_[^_]+_\d{4}_(?P<uid>\d+)\.(?P<ext>tif|json)$",
    re.IGNORECASE,
)
MSK_RX = re.compile(
    r"^(?P<uid>\d+?)_(?P<site_id>\d+)_(?P<date>\d{4}\.\d{2}\.\d{2})_(?P<tag>[A-Za-z]+)"
    r"(_metadata)?\.(?P<ext>tif|json)$",
    re.IGNORECASE,
)

def _std_base(uid: str, site_id: str, date: str) -> str:
    return f"{uid}_{site_id}_{date}"

# strict parse for "<uid>_<site>_<date>_image"
_PARSE_ID_RX = re.compile(
    r"^(?P<uid>\d+?)_(?P<site>\d+)_(?P<date>\d{4}\.\d{2}\.\d{2})_image$"
)

def _parse_std_image_id(file_id: str) -> Tuple[str, str, str]:
    m = _PARSE_ID_RX.match(file_id)
    if not m:
        raise ValueError(f"Unrecognized file_id format: {file_id}")
    return m.group("uid"), m.group("site"), m.group("date")

# ---------------- Helper to ensure JSON-safe serialization ----------------
def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {str(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif hasattr(obj, 'item'):
        return obj.item()
    else:
        return obj


class IrrigationDataSplitter:
    """
    Handles data splitting for irrigation classification with spatial awareness.
    
    Supports the 8-band irrigation label structure:
    - Band 1: Per-pixel irrigation type classification (0-5)
    - Band 2: Per-pixel irrigation presence (0-1)
    - Bands 3-7: Binary uncertainty explanation masks
    - Band 8: Irrigation certainty score (0-4)
    """

    # ---------- NEW: wire in GRIT pairing from Part-1 ----------
    def __init__(self, 
                 csv_path: str,
                 images_dir: str,
                 masks_dir: str,
                 out_root: Optional[str] = None,
                 random_state: int = 42):
        """
        Initialize the data splitter.
        
        Args:
            csv_path: Path to the CSV file with survey data
            images_dir: Directory containing source image files (GRIT features)
            masks_dir: Directory containing source mask/label files (GRIT labels)
            out_root: Root for outputs (optional)
            random_state: Random seed for reproducible splits
        """
        self.csv_path = csv_path
        self.images_dir = str(images_dir)
        self.masks_dir = str(masks_dir)
        self.out_root = str(out_root) if out_root else None
        self.random_state = random_state
        self.df = None
        self.label_encoder = LabelEncoder()

        # map uid -> paired record with true GRIT paths
        self.pairs_by_uid: Dict[str, Dict] = {}

        # Keep your original internal loader name; also provide a public alias
        self._load_data()

    def load_data(self):
        """Public wrapper if you prefer calling the non-underscored method."""
        return self._load_data()

    # ---- Part-1 scan/pair helpers (kept internal; do not affect your public API) ----
    def _scan_images(self, images_dir: str) -> Dict[str, Dict]:
        """
        Scan GRIT features dir and collect files by uid:
        expects 'site_*_*_<year>_<uid>.{tif,json}'
        """
        by_uid = {}
        p = Path(images_dir)
        for fp in p.iterdir():
            if not fp.is_file():
                continue
            m = IMG_RX.match(fp.name)
            if not m:
                continue
            uid = m.group("uid")
            ext = m.group("ext").lower()
            by_uid.setdefault(uid, {"uid": uid, "tif": None, "json": None})
            if ext == "tif":
                by_uid[uid]["tif"] = fp
            else:
                by_uid[uid]["json"] = fp
        return by_uid

    def _scan_masks(self, masks_dir: str) -> Dict[str, Dict]:
        """
        Scan GRIT labels dir and collect mask/json by uid with site/date:
        expects '<uid>_<site>_<YYYY.MM.DD>_<tag>[ _metadata].{tif,json}'
        """
        by_uid = {}
        p = Path(masks_dir)
        for fp in p.iterdir():
            if not fp.is_file():
                continue
            m = MSK_RX.match(fp.name)
            if not m:
                continue
            uid = m.group("uid")
            site_id = m.group("site_id")
            date = m.group("date")
            ext = m.group("ext").lower()
            rec = by_uid.setdefault(uid, {"uid": uid, "site_id": site_id, "date": date, "tif": None, "json": None})
            if ext == "tif":
                rec["tif"] = fp
            else:
                rec["json"] = fp
        # only complete mask pairs
        return {u: r for u, r in by_uid.items() if r["tif"] is not None and r["json"] is not None}

    def _pair_records(self) -> Dict[str, Dict]:
        """
        Pair images and masks by uid, producing standardized base:
        '<uid>_<site>_<YYYY.MM.DD>'.
        """
        imgs = self._scan_images(self.images_dir)
        msks = self._scan_masks(self.masks_dir)
        pairs = {}
        for uid, img in imgs.items():
            if not img.get("tif") or not img.get("json"):
                continue
            m = msks.get(uid)
            if not m or not m.get("tif") or not m.get("json"):
                continue
            base = _std_base(uid, m["site_id"], m["date"])
            pairs[uid] = {
                "uid": uid,
                "site_id": m["site_id"],
                "date": m["date"],
                "base": base,  # "<uid>_<site>_<date>"
                "image_tif": img["tif"],
                "image_json": img["json"],
                "mask_tif": m["tif"],
                "mask_json": m["json"],
            }
        logger.info(f"Paired {len(pairs)} uid(s) across IMAGES/MASKS")
        return pairs

    def _load_data(self):
        """Load and prepare the survey data."""
        self.df = pd.read_csv(self.csv_path)
        
        # Create unique location identifier
        self.df['location_id'] = self.df.apply(
            lambda row: f"{row['y']:.2f}_{row['x']:.2f}", axis=1
        )

        # Use real GRIT scan/pair and keep only rows we can serve
        self.pairs_by_uid = self._pair_records()

        keep_mask = []
        dropped = 0
        for _, row in self.df.iterrows():
            uid = str(row['unique_id']).strip()
            if uid in self.pairs_by_uid:
                keep_mask.append(True)
            else:
                keep_mask.append(False)
                dropped += 1
        if dropped:
            logger.warning(f"Dropped {dropped} CSV rows with no paired GRIT files by uid.")
        self.df = self.df[keep_mask].copy()

        logger.info(f"Final dataset size after file filtering: {len(self.df)}")

    def _get_date_string(self, row):
        """Generate date string from row data in format YYYY.MM.DD."""
        return f"{int(row['year']):04d}.{int(row['month']):02d}.{int(row['day']):02d}"

    def get_band_info(self) -> Dict:
        """
        Get information about the 8-band irrigation label structure.
        
        Returns:
            Dictionary with band descriptions and value ranges
        """
        return {
            "band_1": {
                "name": "Per-pixel irrigation type classification",
                "type": "Categorical",
                "values": {
                    0: "No irrigation",
                    1: "Small-scale",
                    2: "Tree crop", 
                    3: "Industrial",
                    4: "Lawn",
                    5: "Covered"
                }
            },
            "band_2": {
                "name": "Per-pixel irrigation presence",
                "type": "Binary",
                "values": {
                    0: "No irrigation",
                    1: "Irrigation"
                }
            },
            "band_3": {
                "name": "Unclear signs of agriculture",
                "type": "Binary uncertainty mask",
                "values": {0: "Clear", 1: "Unclear"}
            },
            "band_4": {
                "name": "Only slightly green",
                "type": "Binary uncertainty mask", 
                "values": {0: "Green enough", 1: "Slightly green"}
            },
            "band_5": {
                "name": "Uneven",
                "type": "Binary uncertainty mask",
                "values": {0: "Even", 1: "Uneven"}
            },
            "band_6": {
                "name": "May naturally be green",
                "type": "Binary uncertainty mask",
                "values": {0: "Not naturally green", 1: "May be naturally green"}
            },
            "band_7": {
                "name": "May be a fishpond",
                "type": "Binary uncertainty mask",
                "values": {0: "Not fishpond", 1: "May be fishpond"}
            },
            "band_8": {
                "name": "Irrigation certainty score",
                "type": "Categorical",
                "values": {
                    0: "No irrigation",
                    1: "Probably not irrigated",
                    2: "Probably not irrigated", 
                    3: "May be irrigated",
                    4: "Probably irrigated"
                }
            }
        }

    def _get_location_labels_and_files(self, stratification_band: int = 2):
        """
        Get location-level labels and file IDs for stratification.
        
        Args:
            stratification_band: Which band to use for stratification (1-8)
            
        Returns:
            Tuple of (location_labels, location_files) as numpy arrays
        """
        unique_locations = self.df['location_id'].unique()
        location_labels = []
        location_files = []

        for loc_id in unique_locations:
            loc_data = self.df[self.df['location_id'] == loc_id]
            primary_survey = loc_data.iloc[0]
            uid = str(primary_survey['unique_id']).strip()

            pair = self.pairs_by_uid.get(uid)
            if not pair:
                continue  # safety

            file_base = pair['base']  # "<uid>_<site>_<date>"
            file_id = f"{file_base}_image"

            # Use irrigation presence (band 2) for stratification by default
            if stratification_band == 2:
                label = int(primary_survey['irrigation'])  # Binary irrigation presence
            else:
                # For other bands, currently re-use survey 'irrigation' (extend later if needed)
                label = int(primary_survey['irrigation'])

            location_labels.append(label)
            location_files.append(file_id)
        
        return np.array(location_labels), np.array(location_files)

    def spatial_stratified_split(self,
                                test_size: float = 0.2,
                                val_size: float = 0.2,
                                stratification_band: int = 2,
                                min_samples_per_class: int = 5,
                                random_split_only: bool = False) -> Dict:
        """
        Perform spatial stratified split by location.
        
        Args:
            test_size: Proportion of locations for test set
            val_size: Proportion of remaining locations for validation
            stratification_band: Which band to use for stratification (1-8)
            min_samples_per_class: Minimum samples per class for stratification
            random_split_only: If True, performs a basic random split (no stratification or class balance).
            
        Returns:
            Dictionary with train/val/test file lists and metadata
        """
        # Option for fast, non-stratified random split for quick pipeline testing
        location_labels, location_files = self._get_location_labels_and_files(stratification_band)

        if random_split_only:
            logger.info("Performing basic random split (no stratification or class balance).")
            total = len(location_files)
            if total < 3:
                logger.warning(f"Warning: Only {total} locations available. Using all for training. No test/val splits possible.")
                metadata = {
                    'total_locations': int(total),
                    'train_locations': int(total),
                    'val_locations': 0,
                    'test_locations': 0,
                    'stratification_band': int(stratification_band),
                    'class_distribution': {
                        'train': {},
                        'val': {},
                        'test': {}
                    },
                    'warning': 'Insufficient samples for proper train/val/test split'
                }
                return {
                    'train_files': location_files.tolist() if total > 0 else [],
                    'val_files': [],
                    'test_files': [],
                    'metadata': convert_numpy_types(metadata)
                }
            # random split
            train_files, test_files = train_test_split(
                location_files, test_size=test_size, random_state=self.random_state, shuffle=True
            )
            if len(train_files) > 1:
                train_files, val_files = train_test_split(
                    train_files, test_size=val_size / (1 - test_size), random_state=self.random_state, shuffle=True
                )
            else:
                val_files = np.array([])
            metadata = {
                'total_locations': int(len(location_files)),
                'train_locations': int(len(train_files)),
                'val_locations': int(len(val_files)),
                'test_locations': int(len(test_files)),
                'stratification_band': int(stratification_band),
                'class_distribution': {
                    'train': {},
                    'val': {},
                    'test': {}
                },
                'note': 'Random split only, no stratification or class balance'
            }
            return {
                'train_files': train_files.tolist(),
                'val_files': val_files.tolist(),
                'test_files': test_files.tolist(),
                'metadata': convert_numpy_types(metadata)
            }

        # stratified path
        unique_labels, counts = np.unique(location_labels, return_counts=True)
        logger.info(f"Class distribution in {len(location_labels)} locations: " +
                    ", ".join([f"{int(k)}:{int(v)}" for k, v in zip(unique_labels, counts)]))

        if len(location_labels) <= 3 and min_samples_per_class > 1:
            logger.warning(
                f"Very few locations ({len(location_labels)}); lowering min_samples_per_class to 1."
            )
            min_samples_per_class = 1

        valid_classes = unique_labels[counts >= min_samples_per_class]
        valid_mask = np.isin(location_labels, valid_classes)

        if not valid_mask.all():
            logger.warning(f"Warning: {np.sum(~valid_mask)} locations removed due to insufficient class samples (min={min_samples_per_class})")
            location_labels = location_labels[valid_mask]
            location_files = location_files[valid_mask]

        if len(location_files) < 3:
            logger.warning(f"Warning: Only {len(location_files)} locations available. Using all for training. No test/val splits possible.")
            metadata = {
                'total_locations': int(len(location_files)),
                'train_locations': int(len(location_files)),
                'val_locations': 0,
                'test_locations': 0,
                'stratification_band': int(stratification_band),
                'class_distribution': {
                    'train': {str(k): int(v) for k, v in zip(*np.unique(location_labels, return_counts=True))}
                },
                'warning': 'Insufficient samples for proper train/val/test split'
            }
            return {
                'train_files': location_files.tolist() if len(location_files) > 0 else [],
                'val_files': [],
                'test_files': [],
                'metadata': convert_numpy_types(metadata)
            }

        try:
            train_locations, test_locations, train_labels, test_labels = train_test_split(
                location_files, location_labels,
                test_size=test_size,
                stratify=location_labels,
                random_state=self.random_state
            )
        except ValueError as e:
            logger.warning(f"Stratified split failed due to: {e}. Falling back to random split without stratification.")
            train_locations, test_locations, train_labels, test_labels = train_test_split(
                location_files, location_labels,
                test_size=test_size,
                stratify=None,
                random_state=self.random_state
            )

        # Split train into train/val
        try:
            train_locations, val_locations, train_labels, val_labels = train_test_split(
                train_locations, train_labels,
                test_size=val_size / (1 - test_size),
                stratify=train_labels,
                random_state=self.random_state
            )
        except ValueError as e:
            logger.warning(f"Stratified val split failed due to: {e}. Falling back to random split without stratification.")
            train_locations, val_locations, train_labels, val_labels = train_test_split(
                train_locations, train_labels,
                test_size=val_size / (1 - test_size),
                stratify=None,
                random_state=self.random_state
            )

        def _dist(y):
            if len(y) == 0:
                return {}
            k, v = np.unique(y, return_counts=True)
            return {str(int(a)): int(b) for a, b in zip(k, v)}

        metadata = {
            'total_locations': int(len(location_files)),
            'train_locations': int(len(train_locations)),
            'val_locations': int(len(val_locations)),
            'test_locations': int(len(test_locations)),
            'stratification_band': int(stratification_band),
            'class_distribution': {
                'train': _dist(train_labels),
                'val': _dist(val_labels),
                'test': _dist(test_labels)
            }
        }
        return {
            'train_files': train_locations.tolist(),
            'val_files': val_locations.tolist(),
            'test_files': test_locations.tolist(),
            'metadata': convert_numpy_types(metadata)
        }

    def cross_validation_split(self,
                              n_splits: int = 5,
                              stratification_band: int = 2) -> List[Dict]:
        """
        Perform k-fold cross-validation with spatial awareness.
        
        Args:
            n_splits: Number of CV folds
            stratification_band: Which band to use for stratification
            
        Returns:
            List of dictionaries, each containing train/val splits for one fold
        """
        location_labels, location_files = self._get_location_labels_and_files(stratification_band)
        
        if len(location_files) < n_splits:
            logger.warning(f"Insufficient samples ({len(location_files)}) for {n_splits}-fold CV. Using all samples for training.")
            metadata = {
                'train_locations': int(len(location_files)),
                'val_locations': 0,
                'stratification_band': int(stratification_band),
                'class_distribution': {
                    'train': {str(k): int(v) for k, v in zip(*np.unique(location_labels, return_counts=True))},
                    'val': {}
                },
                'warning': f'Insufficient samples for {n_splits}-fold CV'
            }
            return [{
                'fold': 1,
                'train_files': location_files.tolist(),
                'val_files': [],
                'metadata': convert_numpy_types(metadata)
            }]
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        cv_splits = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(location_files, location_labels), 1):
            train_files = location_files[train_idx].tolist()
            val_files = location_files[val_idx].tolist()
            class_distribution = {
                'train': {str(k): int(v) for k, v in zip(*np.unique(location_labels[train_idx], return_counts=True))},
                'val': {str(k): int(v) for k, v in zip(*np.unique(location_labels[val_idx], return_counts=True))}
            }
            metadata = {
                'train_locations': int(len(train_files)),
                'val_locations': int(len(val_files)),
                'stratification_band': int(stratification_band),
                'class_distribution': class_distribution
            }
            cv_splits.append({
                'fold': fold,
                'train_files': train_files,
                'val_files': val_files,
                'metadata': convert_numpy_types(metadata)
            })
        return cv_splits

    def experiment_with_bands(self,
                             target_bands: List[int] = [1, 2, 8],
                             test_size: float = 0.2) -> Dict:
        """
        Create experimental splits for different target bands.
        
        Args:
            target_bands: List of bands to experiment with
            test_size: Test set size
            
        Returns:
            Dictionary with splits for each target band
        """
        experiments = {}
        for band in target_bands:
            logger.info(f"\nCreating split for Band {band}")
            # use irrigation presence as proxy for all bands (extend later)
            split_info = self.spatial_stratified_split(
                test_size=test_size,
                stratification_band=band
            )
            split_info = convert_numpy_types(split_info)
            experiments[f'band_{band}'] = {
                'split_info': split_info,
                'band_description': self.get_band_info()[f'band_{band}'],
                'recommended_use': self._get_band_recommendation(band)
            }
        return experiments

    def _get_band_recommendation(self, band: int) -> str:
        """Get recommendation for how to use a specific band."""
        recommendations = {
            1: "Multi-class classification (6 classes: no irrigation, small-scale, tree crop, industrial, lawn, covered)",
            2: "Binary classification (irrigated vs non-irrigated) - most common use case",
            3: "Binary classification with uncertainty filtering",
            4: "Binary classification with uncertainty filtering", 
            5: "Binary classification with uncertainty filtering",
            6: "Binary classification with uncertainty filtering",
            7: "Binary classification with uncertainty filtering",
            8: "Multi-class classification with confidence levels (5 classes: no irrigation to probably irrigated)"
        }
        return recommendations.get(band, "Unknown band")

    def visualize_splits(self, split_info: Dict, save_path: Optional[str] = None):
        """Visualize the data splits and class distributions."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Class distribution
        metadata = split_info['metadata']
        splits = ['train', 'val', 'test']
        
        for i, split in enumerate(splits):
            if split in metadata['class_distribution']:
                dist = metadata['class_distribution'][split]
                axes[0, 0].bar([f"{split}_{k}" for k in dist.keys()], dist.values(), 
                              alpha=0.7, label=split)
        
        axes[0, 0].set_title('Class Distribution by Split')
        axes[0, 0].set_ylabel('Number of Locations')
        axes[0, 0].legend()
        
        split_sizes = [metadata['train_locations'], metadata['val_locations'], metadata['test_locations']]
        axes[0, 1].pie(split_sizes, labels=['Train', 'Val', 'Test'], autopct='%1.1f%%')
        axes[0, 1].set_title('Split Proportions')
        
        # Spatial distribution
        if 'spatial_info' in metadata:
            pass  # extend if you add spatial viz
        
        summary_text = f"""
        Total Locations: {metadata['total_locations']}
        Train: {metadata['train_locations']} ({metadata['train_locations']/metadata['total_locations']:.1%})
        Val: {metadata['val_locations']} ({metadata['val_locations']/metadata['total_locations']:.1%})
        Test: {metadata['test_locations']} ({metadata['test_locations']/metadata['total_locations']:.1%})
        Stratification Band: {metadata['stratification_band']}
        """
        axes[1, 0].text(0.1, 0.5, summary_text, transform=axes[1, 0].transAxes, 
                       fontsize=12, verticalalignment='center')
        axes[1, 0].set_title('Split Summary')
        axes[1, 0].axis('off')
        
        # Band information
        band_info = self.get_band_info()
        band_text = f"Band {metadata['stratification_band']}:\n"
        band_text += band_info[f'band_{metadata["stratification_band"]}']['name']
        axes[1, 1].text(0.1, 0.5, band_text, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='center')
        axes[1, 1].set_title('Band Information')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def save_splits(self, split_info: Dict, output_dir: str, name: str = "default"):
        """Save split information to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        for split_type in ['train', 'val', 'test']:
            if f'{split_type}_files' in split_info:
                output_path = os.path.join(output_dir, f"{name}_{split_type}_files.json")
                with open(output_path, 'w') as f:
                    json.dump(split_info[f'{split_type}_files'], f, indent=2)
        
        # Save metadata - convert numpy types to Python types for JSON serialization
        metadata = convert_numpy_types(split_info['metadata'])
        metadata_path = os.path.join(output_dir, f"{name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved splits to {output_dir}")

    def _copy_or_link(self, src: str, dst: str, mode: str = "copy"):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(src):
            logger.warning(f"Missing source: {src}")
            return
        if mode == "copy":
            import shutil
            shutil.copy2(src, dst)
        elif mode == "symlink":
            if os.path.lexists(dst):
                os.remove(dst)
            os.symlink(os.path.abspath(src), dst)
        elif mode == "hardlink":
            try:
                os.link(src, dst)
            except OSError:
                import shutil
                shutil.copy2(src, dst)

    def _copy_files_to_fold(self, file_list: List[str], target_dir: str, mode: str = "copy"):
        """Helper method to copy files to a fold directory."""
        for file_id in file_list:
            # file_id is "<uid>_<site>_<date>_image"
            try:
                uid, site, date = _parse_std_image_id(file_id)
            except ValueError as e:
                logger.warning(str(e))
                continue
            base = _std_base(uid, site, date)

            pair = self.pairs_by_uid.get(uid)
            if not pair:
                logger.warning(f"Source pair not found for uid={uid}")
                continue

            # Destinations with standardized names and 'mask' tag
            img_tif_dst  = os.path.join(target_dir, f"{base}_image.tif")
            img_json_dst = os.path.join(target_dir, f"{base}_image.json")
            msk_tif_dst  = os.path.join(target_dir, f"{base}_mask.tif")
            msk_json_dst = os.path.join(target_dir, f"{base}_mask.json")

            self._copy_or_link(str(pair["image_tif"]),  img_tif_dst,  mode=mode)
            self._copy_or_link(str(pair["image_json"]), img_json_dst, mode=mode)
            self._copy_or_link(str(pair["mask_tif"]),   msk_tif_dst,  mode=mode)
            self._copy_or_link(str(pair["mask_json"]),  msk_json_dst, mode=mode)

    def create_folder_structure(self, split_info: Dict, output_dir: str, 
                              copy_files: bool = True, create_symlinks: bool = False) -> str:
        """
        Create train/val/test folder structure and organize files.
        
        Args:
            split_info: Dictionary containing train/val/test file lists
            output_dir: Directory to create the folder structure in
            copy_files: If True, copy files to new structure. If False, create symlinks
            create_symlinks: If True, create symlinks instead of copying (only if copy_files=False)
            
        Returns:
            Path to the created folder structure
        """
        os.makedirs(output_dir, exist_ok=True)
        mode = "copy" if copy_files else ("symlink" if create_symlinks else "copy")

        # Create train/val/test subdirectories
        for split_type in ['train', 'val', 'test']:
            split_dir = os.path.join(output_dir, split_type)
            os.makedirs(split_dir, exist_ok=True)
            
            if f'{split_type}_files' in split_info:
                files = split_info[f'{split_type}_files']
                logger.info(f"Processing {len(files)} files for {split_type} set...")
                self._copy_files_to_fold(files, split_dir, mode=mode)
        
        logger.info(f"Created folder structure at: {output_dir}")
        logger.info(f"  - train/: {len(split_info.get('train_files', []))} files")
        logger.info(f"  - val/: {len(split_info.get('val_files', []))} files")
        logger.info(f"  - test/: {len(split_info.get('test_files', []))} files")
        
        return output_dir

    def save_splits_with_structure(self, split_info: Dict, output_dir: str, 
                                  name: str = "default", copy_files: bool = True) -> str:
        """
        Save split information and create folder structure.
        
        Args:
            split_info: Dictionary containing train/val/test file lists
            output_dir: Directory to save splits and create structure
            name: Name prefix for saved files
            copy_files: If True, copy files to new structure. If False, create symlinks
            
        Returns:
            Path to the created folder structure
        """
        self.save_splits(split_info, output_dir, name)
        structure_dir = os.path.join(output_dir, f"{name}_structure")
        return self.create_folder_structure(split_info, structure_dir, copy_files=copy_files)

    def prepare_experiment_splits(self, 
                                 exp_cfg: Dict,
                                 experiment_dir: str = None) -> Tuple[List[str], List[str], Dict]:
        """
        Prepare data splits for an experiment based on configuration.
        
        Args:
            exp_cfg: Experiment configuration dictionary
            experiment_dir: Directory to save split metadata (optional)
            
        Returns:
            Tuple of (train_files, val_files, split_metadata)
        """
        # Get split parameters from config
        test_size = exp_cfg["data"].get("test_size", 0.2)
        val_size = exp_cfg["data"].get("val_size", 0.2)
        stratification_band = exp_cfg["data"].get("stratification_band", 2)
        min_samples_per_class = exp_cfg["data"].get("min_samples_per_class", 5)
        
        logger.info(f"Creating spatial stratified split:")
        logger.info(f"  - Test size: {test_size}")
        logger.info(f"  - Val size: {val_size}")
        logger.info(f"  - Stratification band: {stratification_band}")
        logger.info(f"  - Min samples per class: {min_samples_per_class}")
        
        split_info = self.spatial_stratified_split(
            test_size=test_size,
            val_size=val_size,
            stratification_band=stratification_band,
            min_samples_per_class=min_samples_per_class
        )
        split_info = convert_numpy_types(split_info)
        
        if exp_cfg["data"].get("create_folder_structure", True):
            splits_dir = exp_cfg["data"].get("splits_dir", "./splits")
            structure_name = exp_cfg["name"]
            copy_files = exp_cfg["data"].get("copy_files", False)
            
            logger.info(f"Creating folder structure at: {splits_dir}")
            structure_path = self.save_splits_with_structure(
                split_info, 
                splits_dir, 
                structure_name,
                copy_files=copy_files
            )
            logger.info(f"Folder structure created at: {structure_path}")
        
        # Prepare split metadata
        split_metadata = {
            "split_info": split_info,
            "splitter_params": convert_numpy_types({
                "test_size": test_size,
                "val_size": val_size,
                "stratification_band": stratification_band,
                "min_samples_per_class": min_samples_per_class
            })
        }
        
        train_files = split_info["train_files"]
        val_files = split_info["val_files"]
        
        logger.info(f"Split created:")
        logger.info(f"  - Train files: {len(train_files)}")
        logger.info(f"  - Val files: {len(val_files)}")
        logger.info(f"  - Test files: {len(split_info['test_files'])}")
        
        return train_files, val_files, split_metadata

    def create_cv_folder_structure(self, 
                                  n_splits: int = 5,
                                  output_dir: str = "./splits",
                                  structure_name: str = "cv_structure",
                                  copy_files: bool = False) -> str:
        """
        Create cross-validation folder structure with file lists.
        
        Args:
            n_splits: Number of CV folds
            output_dir: Directory to save the structure
            structure_name: Name for the structure
            copy_files: If True, copy files. If False, create symlinks or file lists
            
        Returns:
            Path to the created structure
        """
        cv_splits = self.cross_validation_split(n_splits=n_splits)
        
        # Create main output directory
        cv_dir = os.path.join(output_dir, f"{structure_name}")
        os.makedirs(cv_dir, exist_ok=True)
        
        # Get all available files (by uid present after filtering)
        all_files = []
        for loc_id in self.df['location_id'].unique():
            loc_data = self.df[self.df['location_id'] == loc_id]
            primary_survey = loc_data.iloc[0]
            uid = str(primary_survey['unique_id']).strip()
            pair = self.pairs_by_uid.get(uid)
            if not pair:
                continue
            file_id = f"{pair['base']}_image"
            all_files.append(file_id)
        
        # Split into train and test (holdout)
        if len(all_files) < 2:
            logger.warning(f"Insufficient samples ({len(all_files)}) for train/test split. Using all data for training.")
            train_files = all_files
            test_files = []
        else:
            test_size = 0.2
            train_files, test_files = train_test_split(
                all_files, 
                test_size=test_size, 
                random_state=self.random_state
            )
        
        # Save test files list
        test_dir = os.path.join(cv_dir, "test")
        os.makedirs(test_dir, exist_ok=True)
        with open(os.path.join(test_dir, "test_files.txt"), 'w') as f:
            for file_id in test_files:
                f.write(f"{file_id}\n")
        
        # Create CV folds for training data
        train_dir = os.path.join(cv_dir, "train")
        os.makedirs(train_dir, exist_ok=True)
        
        for fold_idx, split_info in enumerate(cv_splits, 1):
            fold_dir = os.path.join(train_dir, f"fold_{fold_idx}")
            os.makedirs(fold_dir, exist_ok=True)
            
            # Create inner train/val directories
            inner_train_dir = os.path.join(fold_dir, "inner_train")
            inner_val_dir = os.path.join(fold_dir, "inner_val")
            os.makedirs(inner_train_dir, exist_ok=True)
            os.makedirs(inner_val_dir, exist_ok=True)
            
            # Save file lists (not copy files)
            with open(os.path.join(inner_train_dir, "train_files.txt"), 'w') as f:
                for file_id in split_info['train_files']:
                    f.write(f"{file_id}\n")
            with open(os.path.join(inner_val_dir, "val_files.txt"), 'w') as f:
                for file_id in split_info['val_files']:
                    f.write(f"{file_id}\n")
            
            # If copy_files is True, actually copy the files
            if copy_files:
                logger.info(f"Copying files for fold {fold_idx}...")
                self._copy_files_to_fold(split_info['train_files'], inner_train_dir, mode="copy")
                self._copy_files_to_fold(split_info['val_files'], inner_val_dir, mode="copy")
        
        # Save metadata
        metadata = {
            "n_splits": n_splits,
            "test_size": len(test_files) / len(all_files) if all_files else 0,
            "total_files": len(all_files),
            "train_files": len(train_files),
            "test_files": len(test_files),
            "cv_splits": convert_numpy_types(cv_splits)
        }
        with open(os.path.join(cv_dir, "cv_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Created CV structure at: {cv_dir}")
        logger.info(f"  - {n_splits} folds created")
        logger.info(f"  - Test set: {len(test_files)} files")
        logger.info(f"  - Train set: {len(train_files)} files")
        
        return cv_dir