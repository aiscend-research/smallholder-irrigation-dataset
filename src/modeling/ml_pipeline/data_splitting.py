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
        
        # Load and prepare data
        self._load_data()
        
    def _load_data(self):
        """Load and prepare the survey data."""
        self.df = pd.read_csv(self.csv_path)
        
        # Create unique location identifier
        self.df['location_id'] = self.df.apply(
            lambda row: f"{row['y']:.2f}_{row['x']:.2f}", axis=1
        )
        
        # Check which files exist in the data directory
        self._validate_data_files()
        
    def _validate_data_files(self):
        """Validate that downloaded files exist for each survey location."""
        existing_files = []
        missing_locations = []
        
        for _, row in self.df.iterrows():
            # Extract site_id number from the CSV (e.g., "id_5168346" -> "5168346")
            site_id_number = row['site_id'].replace('id_', '')
            
            # Use the new naming convention: {unique_id}_{site_id}_{date}_{type}.tif
            # For now, we'll use a placeholder date since we don't have the exact date
            image_filename = f"{row['unique_id']}_{site_id_number}_2023.09.06_image.tif"
            json_filename = f"{row['unique_id']}_{site_id_number}_2023.09.06_image.json"
            
            tif_path = os.path.join(self.data_dir, image_filename)
            json_path = os.path.join(self.data_dir, json_filename)
            
            if os.path.exists(tif_path) and os.path.exists(json_path):
                existing_files.append(image_filename.replace('.tif', ''))
            else:
                missing_locations.append(image_filename)
        
        logger.info(f"Found {len(existing_files)} complete data files")
        logger.warning(f"Missing {len(missing_locations)} data files")
        
        # Filter to only include locations with complete data
        self.df = self.df[self.df.apply(
            lambda row: os.path.exists(os.path.join(
                self.data_dir, 
                f"{row['unique_id']}_{row['site_id'].replace('id_', '')}_2023.09.06_image.tif"
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
        # Get unique locations
        unique_locations = self.df['location_id'].unique()
        
        # Create location-level labels for stratification
        location_labels = []
        location_files = []
        
        for loc_id in unique_locations:
            loc_data = self.df[self.df['location_id'] == loc_id]
            
            # For now, use the first survey's irrigation status
            # In practice, you might want to aggregate across multiple surveys
            primary_survey = loc_data.iloc[0]
            
            # Use the new naming convention: {unique_id}_{site_id}_{date}_{type}
            site_id_number = primary_survey['site_id'].replace('id_', '')
            site_id = f"{primary_survey['unique_id']}_{site_id_number}_2023.09.06_image"
            
            # Use irrigation presence (band 2) for stratification by default
            # This can be modified based on your specific needs
            if stratification_band == 2:
                label = primary_survey['irrigation']  # Binary irrigation presence
            else:
                # For other bands, you'd need to load the actual label files
                # For now, use irrigation as proxy
                label = primary_survey['irrigation']
            
            location_labels.append(label)
            location_files.append(site_id)
        
        # Convert to arrays
        location_labels = np.array(location_labels)
        location_files = np.array(location_files)
        
        # Check class balance
        unique_labels, counts = np.unique(location_labels, return_counts=True)
        logger.info(f"Class distribution in {len(location_labels)} locations:")
        for label, count in zip(unique_labels, counts):
            logger.info(f"  Class {label}: {count} locations")
        
        # Filter classes with too few samples
        valid_classes = unique_labels[counts >= min_samples_per_class]
        valid_mask = np.isin(location_labels, valid_classes)
        
        if not valid_mask.all():
            logger.warning(f"Warning: {np.sum(~valid_mask)} locations removed due to insufficient class samples")
            location_labels = location_labels[valid_mask]
            location_files = location_files[valid_mask]
        
        # Handle case where we have too few samples for splitting
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
        
        # Perform stratified split
        train_locations, test_locations, train_labels, test_labels = train_test_split(
            location_files, location_labels,
            test_size=test_size,
            stratify=location_labels,
            random_state=self.random_state
        )
        
        # Split train into train/val
        train_locations, val_locations, train_labels, val_labels = train_test_split(
            train_locations, train_labels,
            test_size=val_size / (1 - test_size),  # Adjust for the fact that we already split out test
            stratify=train_labels,
            random_state=self.random_state
        )
        
        # Convert to lists
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
        # Get unique locations
        unique_locations = self.df['location_id'].unique()
        
        # Create location-level labels
        location_labels = []
        location_files = []
        
        for loc_id in unique_locations:
            loc_data = self.df[self.df['location_id'] == loc_id]
            primary_survey = loc_data.iloc[0]
            site_id_number = primary_survey['site_id'].replace('id_', '')
            file_id = f"{primary_survey['unique_id']}_{site_id_number}_2023.09.06_image"
            
            label = primary_survey['irrigation']
            location_labels.append(label)
            location_files.append(file_id)
        
        location_labels = np.array(location_labels)
        location_files = np.array(location_files)
        
        # Handle insufficient samples for CV
        if len(location_files) < n_splits:
            logger.warning(f"Insufficient samples ({len(location_files)}) for {n_splits}-fold CV. Using all samples for training.")
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
                    },
                    'warning': f'Insufficient samples for {n_splits}-fold CV'
                }
            }]
        
        # Perform stratified k-fold
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        
        cv_splits = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(location_files, location_labels)):
            train_files = location_files[train_idx].tolist()
            val_files = location_files[val_idx].tolist()
            
            split_info = {
                'fold': fold + 1,
                'train_files': train_files,
                'val_files': val_files,
                'metadata': {
                    'train_locations': len(train_files),
                    'val_locations': len(val_files),
                    'stratification_band': stratification_band,
                    'class_distribution': {
                        'train': dict(zip(*np.unique(location_labels[train_idx], return_counts=True))),
                        'val': dict(zip(*np.unique(location_labels[val_idx], return_counts=True)))
                    }
                }
            }
            cv_splits.append(split_info)
        
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
            
            # For now, use irrigation presence as proxy for all bands
            # In practice, you'd load the actual label files to get band-specific labels
            split_info = self.spatial_stratified_split(
                test_size=test_size,
                stratification_band=band
            )
            
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
        
        # Split sizes
        split_sizes = [metadata['train_locations'], metadata['val_locations'], metadata['test_locations']]
        axes[0, 1].pie(split_sizes, labels=['Train', 'Val', 'Test'], autopct='%1.1f%%')
        axes[0, 1].set_title('Split Proportions')
        
        # Spatial distribution (if coordinates available)
        if 'spatial_info' in metadata:
            # This would show the spatial distribution of splits
            pass
        
        # Summary statistics
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
        
        # Save file lists
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
        
        logger.info(f"Saved splits to {output_dir}")

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
        # Create main output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create train/val/test subdirectories
        for split_type in ['train', 'val', 'test']:
            split_dir = os.path.join(output_dir, split_type)
            os.makedirs(split_dir, exist_ok=True)
            
            if f'{split_type}_files' in split_info:
                files = split_info[f'{split_type}_files']
                logger.info(f"Processing {len(files)} files for {split_type} set...")
                
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
                                # Just create empty files as placeholders
                                with open(dst, 'w') as f:
                                    f.write(f"# Placeholder for {os.path.basename(src)}")
                        else:
                            logger.warning(f"Warning: Source file not found: {src}")
        
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
        # Save split information
        self.save_splits(split_info, output_dir, name)
        
        # Create folder structure
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
        
        # Create splits
        split_info = self.spatial_stratified_split(
            test_size=test_size,
            val_size=val_size,
            stratification_band=stratification_band,
            min_samples_per_class=min_samples_per_class
        )
        
        # Create folder structure if requested
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
            "splitter_params": {
                "test_size": test_size,
                "val_size": val_size,
                "stratification_band": stratification_band,
                "min_samples_per_class": min_samples_per_class
            }
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
        # Create CV splits
        cv_splits = self.cross_validation_split(n_splits=n_splits)
        
        # Create main output directory
        cv_dir = os.path.join(output_dir, f"{structure_name}")
        os.makedirs(cv_dir, exist_ok=True)
        
        # Get all available files
        all_files = []
        for loc_id in self.df['location_id'].unique():
            loc_data = self.df[self.df['location_id'] == loc_id]
            primary_survey = loc_data.iloc[0]
            site_id_number = primary_survey['site_id'].replace('id_', '')
            file_id = f"{primary_survey['unique_id']}_{site_id_number}_2023.09.06_image"
            all_files.append(file_id)
        
        # Handle insufficient samples for train/test split
        if len(all_files) < 2:
            logger.warning(f"Insufficient samples ({len(all_files)}) for train/test split. Using all data for training.")
            train_files = all_files
            test_files = []
        else:
            # Split into train and test
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
                self._copy_files_to_fold(split_info['train_files'], inner_train_dir)
                self._copy_files_to_fold(split_info['val_files'], inner_val_dir)
        
        # Save metadata
        metadata = {
            "n_splits": n_splits,
            "test_size": len(test_files) / len(all_files) if all_files else 0,
            "total_files": len(all_files),
            "train_files": len(train_files),
            "test_files": len(test_files),
            "cv_splits": cv_splits
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
        
        logger.info(f"Created CV structure at: {cv_dir}")
        logger.info(f"  - {n_splits} folds created")
        logger.info(f"  - Test set: {len(test_files)} files")
        logger.info(f"  - Train set: {len(train_files)} files")
        
        return cv_dir
    
    def _copy_files_to_fold(self, file_list: List[str], target_dir: str):
        """Helper method to copy files to a fold directory."""
        for file_id in file_list:
            # Define source files
            image_src = os.path.join(self.data_dir, f"{file_id}.tif")
            label_src = os.path.join(self.data_dir, f"{file_id.replace('_image', '_label')}.tif")
            json_src = os.path.join(self.data_dir, f"{file_id}.json")
            
            # Define destination files
            image_dst = os.path.join(target_dir, f"{file_id}.tif")
            label_dst = os.path.join(target_dir, f"{file_id.replace('_image', '_label')}.tif")
            json_dst = os.path.join(target_dir, f"{file_id}.json")
            
            # Copy files
            for src, dst in [(image_src, image_dst), (label_src, label_dst), (json_src, json_dst)]:
                if os.path.exists(src):
                    import shutil
                    shutil.copy2(src, dst)
                else:
                    logger.warning(f"Source file not found: {src}")


def main():
    """Example usage of the IrrigationDataSplitter."""
    
    # Initialize splitter
    csv_path = "../../data/labels/labeled_surveys/random_sample/latest_irrigation_table.csv"
    data_dir = "../../data/modeling"
    
    splitter = IrrigationDataSplitter(csv_path, data_dir)
    
    # Show band information
    logger.info("Band Information:")
    for band_name, info in splitter.get_band_info().items():
        logger.info(f"{band_name}: {info['name']} ({info['type']})")
    
    # Create basic split
    logger.info("Creating spatial stratified split...")
    split_info = splitter.spatial_stratified_split(
        test_size=0.2,
        val_size=0.2,
        stratification_band=2  # Use irrigation presence
    )
    
    # Visualize splits
    splitter.visualize_splits(split_info)
    
    # Save splits with folder structure
    logger.info("Creating folder structure...")
    structure_path = splitter.save_splits_with_structure(
        split_info, 
        "splits/", 
        "irrigation_binary",
        copy_files=True  # Set to False to create symlinks instead
    )
    
    logger.info(f"Folder structure created at: {structure_path}")
    
    # Experiment with different bands
    logger.info("Creating experimental splits for different bands...")
    experiments = splitter.experiment_with_bands(target_bands=[1, 2, 8])
    
    for band, exp_info in experiments.items():
        logger.info(f"Band {band}: {exp_info['band_description']['name']}")
        logger.info(f"  Recommendation: {exp_info['recommended_use']}")


if __name__ == "__main__":
    main() 