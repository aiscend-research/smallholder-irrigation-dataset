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
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings


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
        
        print(f"Found {len(existing_files)} complete data files")
        print(f"Missing {len(missing_locations)} data files")
        
        # Filter to only include locations with complete data
        self.df = self.df[self.df.apply(
            lambda row: os.path.exists(os.path.join(
                self.data_dir, 
                f"{row['unique_id']}_{row['site_id'].replace('id_', '')}_2023.09.06_image.tif"
            )), axis=1
        )]
        
        print(f"Final dataset size: {len(self.df)} locations")
        
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
        print(f"Class distribution in {len(location_labels)} locations:")
        for label, count in zip(unique_labels, counts):
            print(f"  Class {label}: {count} locations")
        
        # Filter classes with too few samples
        valid_classes = unique_labels[counts >= min_samples_per_class]
        valid_mask = np.isin(location_labels, valid_classes)
        
        if not valid_mask.all():
            print(f"Warning: {np.sum(~valid_mask)} locations removed due to insufficient class samples")
            location_labels = location_labels[valid_mask]
            location_files = location_files[valid_mask]
        
        # Handle case where we have too few samples for splitting
        if len(location_files) < 3:
            print(f"Warning: Only {len(location_files)} locations available. Using all for training.")
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
            site_id = f"site_{primary_survey['y']:.2f}_{primary_survey['x']:.2f}_{primary_survey['year']}_{primary_survey['unique_id']}"
            
            label = primary_survey['irrigation']
            location_labels.append(label)
            location_files.append(site_id)
        
        location_labels = np.array(location_labels)
        location_files = np.array(location_files)
        
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
            print(f"\nCreating split for Band {band}")
            
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
        
        # Save metadata
        metadata_path = os.path.join(output_dir, f"{name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(split_info['metadata'], f, indent=2)
        
        print(f"Saved splits to {output_dir}")


def main():
    """Example usage of the IrrigationDataSplitter."""
    
    # Initialize splitter
    csv_path = "data/labels/labeled_surveys/random_sample/latest_irrigation_table.csv"
    data_dir = "data/features/"
    
    splitter = IrrigationDataSplitter(csv_path, data_dir)
    
    # Print band information
    print("Band Information:")
    for band_name, info in splitter.get_band_info().items():
        print(f"\n{band_name}: {info['name']}")
        print(f"  Type: {info['type']}")
        print(f"  Values: {info['values']}")
    
    # Create basic split
    print("\n" + "="*50)
    print("Creating spatial stratified split...")
    split_info = splitter.spatial_stratified_split(
        test_size=0.2,
        val_size=0.2,
        stratification_band=2  # Use irrigation presence
    )
    
    # Visualize splits
    splitter.visualize_splits(split_info)
    
    # Save splits
    splitter.save_splits(split_info, "splits/", "irrigation_binary")
    
    # Experiment with different bands
    print("\n" + "="*50)
    print("Creating experimental splits for different bands...")
    experiments = splitter.experiment_with_bands(target_bands=[1, 2, 8])
    
    for band, exp_info in experiments.items():
        print(f"\n{band}: {exp_info['band_description']['name']}")
        print(f"  Recommendation: {exp_info['recommended_use']}")
        print(f"  Train/Val/Test: {exp_info['split_info']['metadata']['train_locations']}/"
              f"{exp_info['split_info']['metadata']['val_locations']}/"
              f"{exp_info['split_info']['metadata']['test_locations']}")


if __name__ == "__main__":
    main() 