#!/usr/bin/env python3
"""
Test script to verify AMP dataset works correctly before training
Run this on Gilbreth to validate your data loading
"""

import torch
import numpy as np
from AMP import AMPDataset

def test_dataset():
    print("Testing AMP Dataset...")
    
    # Test paths - update these to match your actual paths on Gilbreth
    train_path = "/scratch/gilbreth/cortesb/amp/amp_dataset/train"
    
    try:
        # Test dataset creation
        print(f"Testing dataset path: {train_path}")
        ds = AMPDataset(root=train_path, valid=False)
        print(f"Dataset loaded successfully! Found {len(ds)} samples")
        
        if len(ds) == 0:
            print("WARNING: Dataset is empty!")
            return
        
        # Test loading one sample
        print("Testing sample loading...")
        img_path, img, (seg_da, seg_ll) = ds[0]
        
        print(f"✅ Sample loaded successfully!")
        print(f"Image path: {img_path}")
        print(f"Image shape: {img.shape} (should be [3, H, W])")
        print(f"Image dtype: {img.dtype} (should be torch.uint8)")
        print(f"Image min/max: {img.min():.1f}/{img.max():.1f}")
        
        print(f"Drivable area mask shape: {seg_da.shape} (should be [2, 360])")
        print(f"Drivable area mask dtype: {seg_da.dtype}")
        print(f"Drivable area unique values: {torch.unique(seg_da)}")
        
        print(f"Lane mask shape: {seg_ll.shape} (should be [2, 360])")
        print(f"Lane mask dtype: {seg_ll.dtype}")
        print(f"Lane mask unique values: {torch.unique(seg_ll)} (should be [0, 255] for dummy)")
        
        # Check if drivable area mask has actual content
        if torch.sum(seg_da[1]) > 0:  # seg_da[1] is the positive mask
            print(f"✅ Drivable area mask contains data: {torch.sum(seg_da[1])} positive pixels")
        else:
            print("⚠️  WARNING: Drivable area mask is empty - check your label files!")
            
        # Test multiple samples
        print("Testing multiple samples...")
        for i in range(min(3, len(ds))):
            _, img_test, (da_test, ll_test) = ds[i]
            assert img_test.shape[0] == 3, f"Wrong image channels: {img_test.shape}"
            assert da_test.shape[0] == 2, f"Wrong DA mask channels: {da_test.shape}"
            assert ll_test.shape[0] == 2, f"Wrong LL mask channels: {ll_test.shape}"
        
        print("✅ All tests passed! Dataset is ready for training.")
        
    except Exception as e:
        print(f"❌ Error testing dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataset()