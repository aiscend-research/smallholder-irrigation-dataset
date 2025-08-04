#!/usr/bin/env python3
"""
Test script to validate our custom datamodule and data splitting code
"""

import os
import sys
import json

def test_data_splitting():
    """Test the data splitting functionality"""
    print("=" * 50)
    print("Testing Data Splitting...")
    
    try:
        from ml_pipeline.data_splitting import IrrigationDataSplitter
        
        csv_path = "../../data/labels/labeled_surveys/random_sample/latest_irrigation_table.csv"
        data_dir = "../../data/modeling"
        
        splitter = IrrigationDataSplitter(csv_path, data_dir)
        print("Data splitter initialized successfully!")
        
        # Test band info
        band_info = splitter.get_band_info()
        print(f"Band info retrieved: {len(band_info)} bands")
        
        # Test spatial split
        split_info = splitter.spatial_stratified_split(
            test_size=0.2, 
            val_size=0.2, 
            stratification_band=2
        )
        print(f"Spatial split created:")
        print(f"   Train: {len(split_info['train_files'])} files")
        print(f"   Val: {len(split_info['val_files'])} files") 
        print(f"   Test: {len(split_info['test_files'])} files")
        
        return True
        
    except Exception as e:
        print(f"Data splitting test failed: {e}")
        return False

def test_custom_dataset():
    """Test the custom dataset functionality"""
    print("\n" + "=" * 50)
    print("Testing Custom Dataset...")
    
    try:
        from custom_dataset import MultiTemporalCropDataset
        
        data_dir = "../../data/modeling"
        sample_files = ["site_-15.04_26.69_2023_1"]
        
        dataset = MultiTemporalCropDataset(data_dir, sample_files)
        print("Dataset initialized successfully!")
        
        # Test loading a sample
        sample = dataset[0]
        print(f"   Sample loaded successfully!")
        print(f"   Sample keys: {list(sample.keys())}")
        print(f"   Image shape: {sample['image'].shape}")
        print(f"   Mask shape: {sample['mask'].shape}")
        print(f"   Metadata keys: {list(sample['metadata'].keys())}")
        
        # Check image dimensions
        expected_shape = (14, 37, 100, 100)  # Based on JSON metadata
        if sample['image'].shape == expected_shape:
            print(f"Image shape correct: {sample['image'].shape}")
        else:
            print(f"Image shape mismatch: expected {expected_shape}, got {sample['image'].shape}")
        
        return True
        
    except Exception as e:
        print(f"Custom dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_factory():
    """Test the dataset factory functionality"""
    print("\n" + "=" * 50)
    print("Testing Dataset Factory...")
    
    try:
        from ml_pipeline.build_features import get_datasets
        
        data_dir = "../../data/modeling"
        train_files = ["site_-15.04_26.69_2023_1"]
        val_files = ["site_-15.04_26.69_2023_1"]
        
        datasets = get_datasets(
            data_dir=data_dir,
            train_files=train_files,
            val_files=val_files,
            label_bands=[1, 2]
        )
        print("Datasets created successfully!")
        print(f"   Available datasets: {list(datasets.keys())}")
        
        # Test train dataset
        train_dataset = datasets['train_dataset']
        sample = train_dataset[0]
        print(f"   Train sample loaded: {sample['image'].shape}")
        
        # Test validation dataset
        val_dataset = datasets['val_dataset']
        sample = val_dataset[0]
        print(f"   Val sample loaded: {sample['image'].shape}")
        
        return True
        
    except Exception as e:
        print(f"Dataset factory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_build_features():
    """Test the build_features factory function"""
    print("\n" + "=" * 50)
    print("Testing Build Features...")
    
    try:
        from ml_pipeline.build_features import get_datasets
        
        data_dir = "../../data/modeling"
        train_files = ["site_-15.04_26.69_2023_1"]
        
        # Test custom datasets creation
        datasets = get_datasets(
            data_dir=data_dir,
            train_files=train_files,
            val_files=[],
            test_files=[],
            label_bands=[2]
        )
        print("Custom datasets created via factory!")
        print(f"   Available datasets: {list(datasets.keys())}")
        
        return True
        
    except Exception as e:
        print(f"Build features test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("Starting Code Validation Tests...")
    print("Based on actual data files in data/modeling/")
    
    tests = [
        test_data_splitting,
        test_custom_dataset,
        test_dataset_factory,
        test_build_features
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total} tests")
    
    if passed == total:
        print("All tests passed!")
    else:
        print("Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 