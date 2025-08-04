#!/usr/bin/env python3
"""
Integration script that combines data splitting with custom datamodule configuration.

This script:
1. Uses the IrrigationDataSplitter to create proper spatial splits
2. Generates YAML configuration files for different experiments
3. Integrates with the custom datamodule
"""

import os
import json
import yaml
from pathlib import Path
from data_splitting import IrrigationDataSplitter


def create_experiment_configs(split_info: dict, 
                            base_config: dict,
                            output_dir: str = "experiment_configs") -> None:
    """
    Create experiment configuration files from split information.
    
    Args:
        split_info: Dictionary containing train/val/test file lists
        base_config: Base configuration template
        output_dir: Directory to save configuration files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create main experiment config
    config = base_config.copy()
    config['data']['datamodule']['custom_params'] = {
        'train_files': split_info['train_files'],
        'val_files': split_info['val_files'], 
        'test_files': split_info['test_files'],
        'label_bands': [2]  # Default to irrigation presence (band 2)
    }
    
    # Save main config
    main_config_path = os.path.join(output_dir, "experiment_main.yaml")
    with open(main_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"Saved main experiment config to {main_config_path}")
    
    # Create band-specific experiment configs
    for band in [1, 2, 8]:  # Most useful bands for experimentation
        band_config = config.copy()
        band_config['name'] = f"{base_config['name']}_band_{band}"
        
        # Update label bands based on the target band
        if band == 1:
            # Multi-class irrigation type classification
            band_config['data']['datamodule']['custom_params']['label_bands'] = [1]
            band_config['model']['hyperparameters']['random_forest']['n_estimators'] = 200  # More trees for multi-class
        elif band == 2:
            # Binary irrigation presence
            band_config['data']['datamodule']['custom_params']['label_bands'] = [2]
        elif band == 8:
            # Multi-class certainty score
            band_config['data']['datamodule']['custom_params']['label_bands'] = [8]
            band_config['model']['hyperparameters']['random_forest']['n_estimators'] = 200
        
        band_config_path = os.path.join(output_dir, f"experiment_band_{band}.yaml")
        with open(band_config_path, 'w') as f:
            yaml.dump(band_config, f, default_flow_style=False, indent=2)
        
        print(f"Saved band {band} experiment config to {band_config_path}")


def create_cross_validation_configs(cv_splits: list,
                                  base_config: dict,
                                  output_dir: str = "cv_configs") -> None:
    """
    Create cross-validation configuration files.
    
    Args:
        cv_splits: List of cross-validation split dictionaries
        base_config: Base configuration template
        output_dir: Directory to save configuration files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i, split_info in enumerate(cv_splits):
        config = base_config.copy()
        config['name'] = f"{base_config['name']}_cv_fold_{split_info['fold']}"
        
        config['data']['datamodule']['custom_params'] = {
            'train_files': split_info['train_files'],
            'val_files': split_info['val_files'],
            'test_files': [],  # No test set in CV
            'label_bands': [2]  # Default to irrigation presence
        }
        
        config_path = os.path.join(output_dir, f"experiment_cv_fold_{split_info['fold']}.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        print(f"Saved CV fold {split_info['fold']} config to {config_path}")


def create_uncertainty_experiment_configs(split_info: dict,
                                        base_config: dict,
                                        output_dir: str = "uncertainty_configs") -> None:
    """
    Create experiment configs for uncertainty-aware classification.
    
    Args:
        split_info: Dictionary containing train/val/test file lists
        base_config: Base configuration template
        output_dir: Directory to save configuration files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Experiment 1: Filter by uncertainty (only use certain predictions)
    config_certain = base_config.copy()
    config_certain['name'] = f"{base_config['name']}_certain_only"
    config_certain['data']['datamodule']['custom_params'] = {
        'train_files': split_info['train_files'],
        'val_files': split_info['val_files'],
        'test_files': split_info['test_files'],
        'label_bands': [2, 8],  # Irrigation presence + certainty score
        'uncertainty_filter': True,
        'certainty_threshold': 3  # Only use "probably irrigated" or "probably not irrigated"
    }
    
    config_path = os.path.join(output_dir, "experiment_certain_only.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config_certain, f, default_flow_style=False, indent=2)
    
    print(f"Saved certain-only experiment config to {config_path}")
    
    # Experiment 2: Use uncertainty as additional features
    config_uncertainty_features = base_config.copy()
    config_uncertainty_features['name'] = f"{base_config['name']}_uncertainty_features"
    config_uncertainty_features['data']['datamodule']['custom_params'] = {
        'train_files': split_info['train_files'],
        'val_files': split_info['val_files'],
        'test_files': split_info['test_files'],
        'label_bands': [2, 3, 4, 5, 6, 7, 8],  # Irrigation + all uncertainty bands
        'use_uncertainty_features': True
    }
    
    config_path = os.path.join(output_dir, "experiment_uncertainty_features.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config_uncertainty_features, f, default_flow_style=False, indent=2)
    
    print(f"Saved uncertainty features experiment config to {config_path}")


def main():
    """Main function to create all experiment configurations."""
    
    # Paths
    csv_path = "data/labels/labeled_surveys/random_sample/latest_irrigation_table.csv"
    data_dir = "data/features/"
    
    # Initialize splitter
    print("Initializing data splitter...")
    splitter = IrrigationDataSplitter(csv_path, data_dir)
    
    # Base configuration template
    base_config = {
        'name': 'irrigation_classification',
        'data': {
            'dataset_path': data_dir,
            'train_subset_size': None,  # Use all data
            'val_subset_size': None,
            'test_subset_size': None,
            'datamodule': {
                'type': 'custom',
                'batch_size': 4,
                'num_workers': 0,
                'custom_params': {}
            }
        },
        'model': {
            'type': 'random_forest',
            'hyperparameters': {
                'random_forest': {
                    'n_estimators': 100,
                    'random_state': 42,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2
                },
                'gradient_boosting': {
                    'n_estimators': 200,
                    'learning_rate': 0.05,
                    'max_depth': 4,
                    'subsample': 0.8,
                    'min_samples_split': 5,
                    'min_samples_leaf': 3,
                    'random_state': 42
                }
            }
        },
        'visualization': {
            'colors': ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF'],
            'num_samples': 2
        },
        'output': {
            'base_dir': './experiments'
        }
    }
    
    # Create main spatial stratified split
    print("\nCreating spatial stratified split...")
    split_info = splitter.spatial_stratified_split(
        test_size=0.2,
        val_size=0.2,
        stratification_band=2  # Use irrigation presence
    )
    
    # Save split information
    splitter.save_splits(split_info, "splits/", "irrigation_main")
    
    # Create experiment configurations
    print("\nCreating experiment configurations...")
    create_experiment_configs(split_info, base_config, "experiment_configs")
    
    # Create cross-validation configurations
    print("\nCreating cross-validation configurations...")
    cv_splits = splitter.cross_validation_split(n_splits=5, stratification_band=2)
    create_cross_validation_configs(cv_splits, base_config, "cv_configs")
    
    # Create uncertainty experiment configurations
    print("\nCreating uncertainty experiment configurations...")
    create_uncertainty_experiment_configs(split_info, base_config, "uncertainty_configs")
    
    # Create band experimentation configurations
    print("\nCreating band experimentation configurations...")
    experiments = splitter.experiment_with_bands(target_bands=[1, 2, 8], test_size=0.2)
    
    for band_name, exp_info in experiments.items():
        band_config = base_config.copy()
        band_config['name'] = f"irrigation_{band_name}"
        band_config['data']['datamodule']['custom_params'] = {
            'train_files': exp_info['split_info']['train_files'],
            'val_files': exp_info['split_info']['val_files'],
            'test_files': exp_info['split_info']['test_files'],
            'label_bands': [int(band_name.split('_')[1])]
        }
        
        config_path = f"experiment_configs/experiment_{band_name}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(band_config, f, default_flow_style=False, indent=2)
        
        print(f"Saved {band_name} experiment config to {config_path}")
    
    print("\n" + "="*50)
    print("Configuration generation complete!")
    print("\nGenerated configurations:")
    print("- experiment_configs/: Main experiments with different bands")
    print("- cv_configs/: Cross-validation experiments")
    print("- uncertainty_configs/: Uncertainty-aware experiments")
    print("- splits/: Split information and metadata")
    
    print("\nNext steps:")
    print("1. Review the generated configurations")
    print("2. Update dataset_path in configs to point to your data directory")
    print("3. Run experiments: python run_experiment.py experiment_configs/experiment_band_2.yaml")
    print("4. Implement proper label loading in custom_dataset.py")


if __name__ == "__main__":
    main() 