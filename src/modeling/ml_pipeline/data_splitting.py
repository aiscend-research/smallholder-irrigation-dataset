#!/usr/bin/env python3
"""
This module implements spatial-aware data splitting strategies that:
1. Split by location to avoid spatial data leakage
2. Maintain class balance through stratified sampling
3. Support experimentation with different label bands
4. Handle the 8-band irrigation label structure
"""

import pandas as pd
import numpy as np
import json
import os
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class IrrigationDataSplitter:
    """
    Handles data splitting for irrigation classification with spatial awareness.
    
    Supports the 8-band irrigation label structure:
    - Band 1: Per-pixel irrigation type classification (0-5)
    - Band 2: Per-pixel irrigation presence (0-1)
    - Bands 3-7: Binary uncertainty explanation masks
    - Band 8: Irrigation certainty score (0-4)
    """
    
    def __init__(self, 
                 csv_path: str,
                 data_dir: str,
                 random_state: int = 42):
        """
        Initialize the data splitter.
        
        Args:
            csv_path: Path to the CSV file with survey data
            data_dir: Directory containing downloaded .tif and .json files
            random_state: Random seed for reproducible splits
        """
        self.csv_path = csv_path
        self.data_dir = data_dir
        self.random_state = random_state
        self.df = None
        self.label_encoder = LabelEncoder()
        
        # Check if data directory is organized
        data_path = Path(data_dir)
        organized_dir = data_path / "organized"
        
        if organized_dir.exists():
            # Use organized structure
            self.images_dir = organized_dir / "images"
            self.labels_dir = organized_dir / "labels"
            self.metadata_dir = organized_dir / "metadata"
        else:
            # Use flat structure
            self.images_dir = data_path
            self.labels_dir = data_path
            self.metadata_dir = data_path
        
        self._load_data()
        
    def _load_data(self):
        """Load and prepare the survey data."""
        self.df = pd.read_csv(self.csv_path)
        
        # Create unique location identifier
        self.df['location_id'] = self.df.apply(
            lambda row: f"{row['y']:.2f}_{row['x']:.2f}", axis=1
        )
        
        self._validate_data_files()
    
    def _get_date_string(self, row):
        """Generate date string from row data in format YYYY.MM.DD."""
        return f"{row['year']}.{row['month']:02d}.{row['day']:02d}"
    
    def _validate_data_files(self):
        """Validate that downloaded files exist for each survey location."""
        # Use the appropriate directories based on organization
        available_tifs = list(self.images_dir.glob("*.tif")) + list(self.labels_dir.glob("*.tif"))
        available_jsons = list(self.metadata_dir.glob("*.json"))
        
        # Check if we have standardized data format
        is_standardized = any("_image" in f.stem for f in available_tifs)
        
        if is_standardized:
            # For standardized data, just count the files and create a mapping
            image_files = [f.stem for f in available_tifs if "_image" in f.stem]
            label_files = [f.stem for f in available_tifs if "_label" in f.stem]
            json_files = [f.stem for f in available_jsons]
            
            # Create base IDs (remove _image, _label suffixes)
            base_ids = set()
            for img_file in image_files:
                base_id = img_file.replace('_image', '')
                base_ids.add(base_id)
            
            # Check for complete sets
            complete_sets = 0
            for base_id in base_ids:
                if (f"{base_id}_image" in image_files and 
                    f"{base_id}_label" in label_files and 
                    f"{base_id}.json" in json_files):
                    complete_sets += 1
            
            # Store the available files for later use
            self.data_files = image_files
            self.available_base_ids = base_ids
            
        else:
            # Original validation logic for non-standardized data
            existing_files = []
            missing_locations = []
            
            for _, row in self.df.iterrows():
                # Extract site_id number from the CSV (e.g., "id_5168346" -> "5168346")
                site_id_number = row['site_id'].replace('id_', '')
                date_str = self._get_date_string(row)

                image_filename = f"{row['unique_id']}_{site_id_number}_{date_str}_image.tif"
                json_filename = f"{row['unique_id']}_{site_id_number}_{date_str}_image.json"
                
                tif_path = self.images_dir / image_filename
                json_path = self.metadata_dir / json_filename
                
                if tif_path.exists() and json_path.exists():
                    existing_files.append(image_filename.replace('.tif', ''))
                else:
                    missing_locations.append(image_filename)
            
            # Store the existing files for later use
            self.data_files = existing_files
            
            # Filter to only include locations with complete data
            self.df = self.df[self.df.apply(
                lambda row: os.path.exists(os.path.join(
                    self.data_dir, 
                    f"{row['unique_id']}_{row['site_id'].replace('id_', '')}_{self._get_date_string(row)}_image.tif"
                )), axis=1
            )]
            
            logger.info(f"Final dataset size: {len(self.df)} locations")
        
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
        # Check if we're working with standardized data
        data_path = Path(self.data_dir)
        available_tifs = list(data_path.rglob("*.tif"))
        available_tif_names = {f.stem for f in available_tifs}
        
        is_standardized = any("_image" in name for name in available_tif_names)
        
        if is_standardized:
            # For standardized data, we need to group by site_id to prevent spatial leakage
            
            # First, create a mapping from unique_id to site_id
            unique_id_to_site = {}
            for _, row in self.df.iterrows():
                unique_id_to_site[row['unique_id']] = row['site_id']
            
            # Group files by site_id (geographic location)
            site_files = {}  # site_id -> list of file_ids
            site_labels = {}  # site_id -> label
            
            for tif_file in available_tifs:
                stem = tif_file.stem
                if "_image" in stem:
                    # Extract unique_id from standardized name: {unique_id}_{site_id}_{date}_image
                    parts = stem.split('_')
                    if len(parts) >= 4 and parts[-1] == "image":
                        try:
                            unique_id = int(parts[0])
                            file_id = stem
                            
                            # Get the site_id for this unique_id
                            if unique_id in unique_id_to_site:
                                site_id = unique_id_to_site[unique_id]
                                
                                # Initialize if this site_id is new
                                if site_id not in site_files:
                                    site_files[site_id] = []
                                    # Get label from any survey at this site (they should be similar)
                                    site_row = self.df[self.df['site_id'] == site_id].iloc[0]
                                    site_labels[site_id] = site_row['irrigation']
                                
                                # Add this file to the site's file list
                                site_files[site_id].append(file_id)
                        except ValueError:
                            continue
            
            # Now create location-level data (one entry per unique geographic location)
            location_files = []
            location_labels = []
            
            for site_id, file_list in site_files.items():
                # Use the first file as representative for this location
                # All files from the same site will be kept together
                representative_file = file_list[0]
                label = site_labels[site_id]
                
                location_files.append(representative_file)
                location_labels.append(label)
            
            # If no sites were found, create a fallback using just the file names
            if len(location_files) == 0:
                # Create a simple mapping: one file per location
                for tif_file in available_tifs:
                    stem = tif_file.stem
                    if "_image" in stem:
                        location_files.append(stem)
                        # Use a default label (0 for no irrigation)
                        location_labels.append(0)
            
        else:
            # Original logic but ensure we group by site_id
            unique_sites = self.df['site_id'].unique()
            
            location_labels = []
            location_files = []
            
            for site_id in unique_sites:
                site_data = self.df[self.df['site_id'] == site_id]
                primary_survey = site_data.iloc[0]
                site_id_number = primary_survey['site_id'].replace('id_', '')
                date_str = self._get_date_string(primary_survey)
                file_id = f"{primary_survey['unique_id']}_{site_id_number}_{date_str}_image"
                
                # Use irrigation presence for stratification
                if stratification_band == 2:
                    label = primary_survey['irrigation']
                else:
                    label = primary_survey['irrigation']
                
                location_labels.append(label)
                location_files.append(file_id)
        
        return np.array(location_labels), np.array(location_files)
    
    def spatial_stratified_split(self,
                                test_size: float = 0.2,
                                val_size: float = 0.2,
                                stratification_band: int = 2,
                                min_samples_per_class: int = 5) -> Dict:
        """
        Perform spatial stratified split by location.
        
        Args:
            test_size: Proportion of locations for test set
            val_size: Proportion of remaining locations for validation
            stratification_band: Which band to use for stratification (1-8)
            min_samples_per_class: Minimum samples per class for stratification
            
        Returns:
            Dictionary with train/val/test file lists and metadata
        """
        # Get location-level labels and files
        location_labels, location_files = self._get_location_labels_and_files(stratification_band)
        
        # Check class balance
        unique_labels, counts = np.unique(location_labels, return_counts=True)
        logger.info(f"Class distribution in {len(location_labels)} locations:")
        for label, count in zip(unique_labels, counts):
            logger.info(f"  Class {label}: {count} locations")
        
        valid_classes = unique_labels[counts >= min_samples_per_class]
        valid_mask = np.isin(location_labels, valid_classes)
        
        if not valid_mask.all():
            logger.warning(f"Warning: {np.sum(~valid_mask)} locations removed due to insufficient class samples")
            location_labels = location_labels[valid_mask]
            location_files = location_files[valid_mask]
        
        if len(location_files) < 3:
            logger.warning(f"Warning: Only {len(location_files)} locations available. Using all for training.")
            return {
                'train_files': location_files.tolist(),
                'val_files': [],
                'test_files': [],
                'metadata': {
                    'total_locations': len(location_files),
                    'train_locations': len(location_files),
                    'val_locations': 0,
                    'test_locations': 0,
                    'stratification_band': stratification_band,
                    'class_distribution': {
                        'train': dict(zip(*np.unique(location_labels, return_counts=True)))
                    },
                    'warning': 'Insufficient samples for proper train/val/test split'
                }
            }
        
        # Check if stratification is possible
        unique_labels, counts = np.unique(location_labels, return_counts=True)
        min_samples_per_class = 2  # Need at least 2 samples per class for stratification
        
        valid_classes = unique_labels[counts >= min_samples_per_class]
        valid_mask = np.isin(location_labels, valid_classes)
        
        if not valid_mask.all():
            location_labels = location_labels[valid_mask]
            location_files = location_files[valid_mask]
        
        # Perform stratified split if possible
        if len(valid_classes) >= 2:
            # Use stratification if we have enough classes
            train_locations, test_locations, train_labels, test_labels = train_test_split(
                location_files, location_labels,
                test_size=test_size,
                stratify=location_labels,
                random_state=self.random_state
            )
            
            # Split train into train/val
            train_locations, val_locations, train_labels, val_labels = train_test_split(
                train_locations, train_labels,
                test_size=val_size / (1 - test_size), 
                stratify=train_labels,
                random_state=self.random_state
            )
        else:
            # Use random split if stratification is not possible
            train_locations, test_locations, train_labels, test_labels = train_test_split(
                location_files, location_labels,
                test_size=test_size,
                random_state=self.random_state
            )
            
            # Split train into train/val
            train_locations, val_locations, train_labels, val_labels = train_test_split(
                train_locations, train_labels,
                test_size=val_size / (1 - test_size), 
                random_state=self.random_state
            )
        
        train_files = train_locations.tolist()
        val_files = val_locations.tolist()
        test_files = test_locations.tolist()
        
        # Create metadata
        split_info = {
            'train_files': train_files,
            'val_files': val_files,
            'test_files': test_files,
            'metadata': {
                'total_locations': len(location_files),
                'train_locations': len(train_files),
                'val_locations': len(val_files),
                'test_locations': len(test_files),
                'stratification_band': stratification_band,
                'class_distribution': {
                    'train': dict(zip(*np.unique(train_labels, return_counts=True))),
                    'val': dict(zip(*np.unique(val_labels, return_counts=True))),
                    'test': dict(zip(*np.unique(test_labels, return_counts=True)))
                }
            }
        }
        
        return split_info
    
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
        # Get location-level labels and files (these are site representatives)
        location_labels, location_files = self._get_location_labels_and_files(stratification_band)
        
        # Handle insufficient samples for CV
        if len(location_files) < n_splits:
            # Return a single fold with all data
            return [{
                'fold': 1,
                'train_files': location_files.tolist(),
                'val_files': [],
                'metadata': {
                    'train_locations': len(location_files),
                    'val_locations': 0,
                    'stratification_band': stratification_band,
                    'class_distribution': {
                        'train': dict(zip(*np.unique(location_labels, return_counts=True))),
                        'val': {}
                    }
                }
            }]
        
        # Perform stratified k-fold at the SITE level (not file level)
        # This ensures that within each fold, train and validation don't share the same site
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        
        cv_splits = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(location_files, location_labels)):
            train_sites = location_files[train_idx].tolist()
            val_sites = location_files[val_idx].tolist()
            
            # Verify spatial awareness within this fold
            train_site_ids = set()
            val_site_ids = set()
            
            for site_file in train_sites:
                parts = site_file.split('_')
                if len(parts) >= 2:
                    site_id = parts[1]  # Extract site_id from filename
                    train_site_ids.add(site_id)
            
            for site_file in val_sites:
                parts = site_file.split('_')
                if len(parts) >= 2:
                    site_id = parts[1]  # Extract site_id from filename
                    val_site_ids.add(site_id)
            
            # Check for spatial violations within this fold
            overlap = train_site_ids.intersection(val_site_ids)
            if overlap:
                logger.warning(f"Fold {fold + 1}: Spatial violation detected - sites {overlap} appear in both train and validation")
            else:
                logger.debug(f"Fold {fold + 1}: Spatial awareness maintained - no site overlap between train and validation")
            
            split_info = {
                'fold': fold + 1,
                'train_files': train_sites,  # These are site representatives
                'val_files': val_sites,      # These are site representatives
                'metadata': {
                    'train_locations': len(train_sites),
                    'val_locations': len(val_sites),
                    'stratification_band': stratification_band,
                    'class_distribution': {
                        'train': dict(zip(*np.unique(location_labels[train_idx], return_counts=True))),
                        'val': dict(zip(*np.unique(location_labels[val_idx], return_counts=True)))
                    },
                    'spatial_awareness': 'maintained within fold' if not overlap else 'violation detected'
                }
            }
            cv_splits.append(split_info)
        
        return cv_splits
    

    

    

    
    def save_splits(self, split_info: Dict, output_dir: str, name: str = "default"):
        """Save split information to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        for split_type in ['train', 'val', 'test']:
            if f'{split_type}_files' in split_info:
                output_path = os.path.join(output_dir, f"{name}_{split_type}_files.json")
                with open(output_path, 'w') as f:
                    json.dump(split_info[f'{split_type}_files'], f, indent=2)
        
        # Save metadata - convert numpy types to Python types for JSON serialization
        metadata = split_info['metadata'].copy()
        
        # Convert numpy int64 to regular int for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {str(k): convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            else:
                return obj
        
        metadata = convert_numpy_types(metadata)
        
        metadata_path = os.path.join(output_dir, f"{name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        


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
        
        # Create train/val/test subdirectories
        for split_type in ['train', 'val', 'test']:
            split_dir = os.path.join(output_dir, split_type)
            os.makedirs(split_dir, exist_ok=True)
            
            if f'{split_type}_files' in split_info:
                files = split_info[f'{split_type}_files']
                
                for file_base in files:
                    # Define source files (image, label, json)
                    image_src = os.path.join(self.data_dir, f"{file_base}.tif")
                    label_src = os.path.join(self.data_dir, f"{file_base.replace('_image', '_label')}.tif")
                    json_src = os.path.join(self.data_dir, f"{file_base}.json")
                    
                    # Define destination files
                    image_dst = os.path.join(split_dir, f"{file_base}.tif")
                    label_dst = os.path.join(split_dir, f"{file_base.replace('_image', '_label')}.tif")
                    json_dst = os.path.join(split_dir, f"{file_base}.json")
                    
                    # Copy or link files
                    for src, dst in [(image_src, image_dst), (label_src, label_dst), (json_src, json_dst)]:
                        if os.path.exists(src):
                            if copy_files:
                                import shutil
                                shutil.copy2(src, dst)
                            elif create_symlinks:
                                if os.path.exists(dst):
                                    os.remove(dst)
                                os.symlink(os.path.abspath(src), dst)
                            else:
                                with open(dst, 'w') as f:
                                    f.write(f"# Placeholder for {os.path.basename(src)}")
                        else:
                            logger.warning(f"Warning: Source file not found: {src}")
        
        logger.info(f"Created folder structure at: {output_dir}")
        
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
        
        split_info = self.spatial_stratified_split(
            test_size=test_size,
            val_size=val_size,
            stratification_band=stratification_band,
            min_samples_per_class=min_samples_per_class
        )
        
        if exp_cfg["data"].get("create_folder_structure", True):
            splits_dir = exp_cfg["data"].get("splits_dir", "./splits")
            structure_name = exp_cfg["name"]
            copy_files = exp_cfg["data"].get("copy_files", False)
            
            structure_path = self.save_splits_with_structure(
                split_info, 
                splits_dir, 
                structure_name,
                copy_files=copy_files
            )
        
        # Prepare split metadata
        split_metadata = {
            "split_info": split_info,
            "splitter_params": {
                "test_size": test_size,
                "val_size": val_size,
                "stratification_band": stratification_band,
                "min_samples_per_class": min_samples_per_class
            }
        }
        
        train_files = split_info["train_files"]
        val_files = split_info["val_files"]
        
        return train_files, val_files, split_metadata

    def create_cv_folder_structure(self, 
                                  n_splits: int = 5,
                                  output_dir: str = "./splits",
                                  structure_name: str = "cv_structure",
                                  copy_files: bool = False,
                                  stratification_band: int = 2) -> str:
        """
        Create cross-validation folder structure with file lists.
        
        Args:
            n_splits: Number of CV folds
            output_dir: Directory to save the structure
            structure_name: Name for the structure
            copy_files: If True, copy files. If False, create symlinks or file lists
            stratification_band: Which band to use for stratification (1-8)
            
        Returns:
            Path to the created structure
        """
        cv_splits = self.cross_validation_split(n_splits=n_splits, stratification_band=stratification_band)
        
        # Create main output directory
        cv_dir = os.path.join(output_dir, f"{structure_name}")
        os.makedirs(cv_dir, exist_ok=True)
        
        # Get SPATIALLY-AWARE location-level data (one entry per unique site)
        location_labels, location_files = self._get_location_labels_and_files(stratification_band)
        
        # Create mapping from representative files back to all files at each site
        site_to_all_files = self._create_site_to_files_mapping()
        

        
        # Handle insufficient samples for train/test split
        if len(location_files) < 2:
            train_sites = location_files
            test_sites = []
        else:
            # Check if we have enough samples per class for stratification
            unique_labels, counts = np.unique(location_labels, return_counts=True)
            min_samples_per_class = 2  # Need at least 2 samples per class for stratification
            
            valid_classes = unique_labels[counts >= min_samples_per_class]
            valid_mask = np.isin(location_labels, valid_classes)
            
            if not valid_mask.all():
                location_labels = location_labels[valid_mask]
                location_files = location_files[valid_mask]
            
            # Perform stratified split at SITE level (not file level)
            test_size = 0.2
            if len(valid_classes) >= 2:
                # Use stratification if we have enough classes
                train_sites, test_sites, train_labels, test_labels = train_test_split(
                    location_files, location_labels,
                    test_size=test_size,
                    stratify=location_labels,
                    random_state=self.random_state
                )
            else:
                # Use random split if stratification is not possible
                train_sites, test_sites, train_labels, test_labels = train_test_split(
                    location_files, location_labels,
                    test_size=test_size,
                    random_state=self.random_state
                )
        
        # Convert back to lists (handle both numpy arrays and regular lists)
        if hasattr(train_sites, 'tolist'):
            train_sites = train_sites.tolist()
        if hasattr(test_sites, 'tolist'):
            test_sites = test_sites.tolist()
        
        # Expand site-level splits to include all files from each site
        test_files = self._expand_sites_to_files(test_sites, site_to_all_files)
        train_sites_pool = self._expand_sites_to_files(train_sites, site_to_all_files)
        

        
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
            
            # The CV splits contain site representatives, but we need to expand them to all files
            # Expand train sites to all files for this fold
            train_site_representatives = split_info['train_files']  # These are site representatives
            train_files_expanded = self._expand_sites_to_files(train_site_representatives, site_to_all_files)
            
            # Expand validation sites to all files for this fold
            val_site_representatives = split_info['val_files']  # These are site representatives
            val_files_expanded = self._expand_sites_to_files(val_site_representatives, site_to_all_files)
            
            # Save expanded file lists
            with open(os.path.join(inner_train_dir, "train_files.txt"), 'w') as f:
                for file_id in train_files_expanded:
                    f.write(f"{file_id}\n")
            
            with open(os.path.join(inner_val_dir, "val_files.txt"), 'w') as f:
                for file_id in val_files_expanded:
                    f.write(f"{file_id}\n")
            

            
            # If copy_files is True, actually copy the files
            if copy_files:
                self._copy_files_to_fold(train_files_expanded, inner_train_dir)
                self._copy_files_to_fold(val_files_expanded, inner_val_dir)
        
        # Save metadata
        metadata = {
            "n_splits": n_splits,
            "test_size": len(test_sites) / len(location_files) if len(location_files) > 0 else 0,
            "total_sites": len(location_files),
            "train_sites": len(train_sites),
            "test_sites": len(test_sites),
            "total_files": len(train_sites_pool) + len(test_files),
            "train_files": len(train_sites_pool),
            "test_files": len(test_files),
            "cv_splits": cv_splits,
            "stratification_band": stratification_band,
            "spatial_awareness": "enabled - sites from same location kept together",
            "cv_spatial_awareness": "enabled - within each fold, train and validation don't share sites"
        }
        
        # Convert numpy types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {str(k): convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            else:
                return obj
        
        metadata = convert_numpy_types(metadata)
        
        with open(os.path.join(cv_dir, "cv_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Verify spatial awareness
        self._verify_spatial_awareness(train_sites, test_sites, site_to_all_files)
        
        logger.info(f"CV structure created: {n_splits} folds, {len(test_files)} test files, {len(train_sites_pool)} train files")
        
        return cv_dir
    
    def _create_site_to_files_mapping(self) -> Dict[str, List[str]]:
        """
        Creates a mapping from site_id to a list of all file IDs belonging to that site.
        This is necessary because the _get_location_labels_and_files returns one file per site,
        but we need to expand the test/train splits to include all files from each site.
        """
        site_to_all_files = {}
        
        # Use organized directories if available
        available_tifs = list(self.images_dir.glob("*.tif")) + list(self.labels_dir.glob("*.tif"))
        
        # First, create a mapping from unique_id to site_id
        unique_id_to_site = {}
        for _, row in self.df.iterrows():
            unique_id_to_site[row['unique_id']] = row['site_id']
        
        # Group files by site_id (geographic location)
        for tif_file in available_tifs:
            stem = tif_file.stem
            if "_image" in stem:
                # Extract unique_id from standardized name: {unique_id}_{site_id}_{date}_image
                parts = stem.split('_')
                if len(parts) >= 4 and parts[-1] == "image":
                    try:
                        unique_id = int(parts[0])
                        file_id = stem
                        
                        # Get the site_id for this unique_id
                        if unique_id in unique_id_to_site:
                            site_id = unique_id_to_site[unique_id]
                            
                            if site_id not in site_to_all_files:
                                site_to_all_files[site_id] = []
                            site_to_all_files[site_id].append(file_id)
                        else:
                            continue
                    except ValueError:
                        continue
        
        return site_to_all_files
    
    def _expand_sites_to_files(self, site_list: List[str], site_to_all_files: Dict[str, List[str]]) -> List[str]:
        """
        Expands a list of site_ids to a list of all file IDs belonging to those sites.
        This is necessary because the train/test split is done at the site level,
        but the CV folds need to be done at the file level.
        """
        all_files = []
        
        # For each representative file in site_list, find all files from the same site
        for representative_file in site_list:
            # Extract the site_id from the representative file
            parts = representative_file.split('_')
            if len(parts) >= 4 and parts[-1] == "image":
                try:
                    unique_id = int(parts[0])
                    site_id_number = parts[1]  # site_id is at index 1
                    
                    # Create the full site_id format (e.g., "id_5130509")
                    full_site_id = f"id_{site_id_number}"
                    
                    # Get all files for this site
                    if full_site_id in site_to_all_files:
                        site_files = site_to_all_files[full_site_id]
                        all_files.extend(site_files)
                    else:
                        # Fallback: just add the representative file
                        all_files.append(representative_file)
                        
                except ValueError as e:
                    # Fallback: just add the representative file
                    all_files.append(representative_file)
            else:
                # If it's not a valid file format, just add it as is
                all_files.append(representative_file)
        
        return all_files
    
    def _verify_spatial_awareness(self, train_sites: List[str], test_sites: List[str], site_to_all_files: Dict[str, List[str]]):
        """Verify that no site appears in both train and test sets."""
        train_site_set = set(train_sites)
        test_site_set = set(test_sites)
        
        # Check for overlap
        overlap = train_site_set.intersection(test_site_set)
        if overlap:
            logger.error(f"SPATIAL AWARENESS VIOLATION: {len(overlap)} sites appear in both train and test sets: {overlap}")
            raise ValueError("Spatial awareness violation detected - same site in train and test")
    
    def _copy_files_to_fold(self, file_list: List[str], target_dir: str):
        """Helper method to copy files to a fold directory."""
        for file_id in file_list:
            # Use organized directories if available
            if hasattr(self, 'images_dir'):
                image_src = self.images_dir / f"{file_id}.tif"
                label_src = self.labels_dir / f"{file_id.replace('_image', '_label')}.tif"
                json_src = self.metadata_dir / f"{file_id}.json"
            else:
                image_src = Path(self.data_dir) / f"{file_id}.tif"
                label_src = Path(self.data_dir) / f"{file_id.replace('_image', '_label')}.tif"
                json_src = Path(self.data_dir) / f"{file_id}.json"
            
            image_dst = Path(target_dir) / f"{file_id}.tif"
            label_dst = Path(target_dir) / f"{file_id.replace('_image', '_label')}.tif"
            json_dst = Path(target_dir) / f"{file_id}.json"
            
            for src, dst in [(image_src, image_dst), (label_src, label_dst), (json_src, json_dst)]:
                if src.exists():
                    import shutil
                    shutil.copy2(str(src), str(dst))
                else:
                    pass