#!/usr/bin/env python3
"""
Test script to verify all imports work correctly
"""

import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

print("Testing imports...")

try:
    print("✅ Importing config...")
    from src.utils.config import config
    print(f"   Base directory: {config.base_data_dir}")
    
    print("✅ Importing ViT model...")
    from src.hsmodel.vit_pytorch import ViT
    print("   ViT model imported successfully")
    
    print("✅ Importing preprocessing modules...")
    import src.preprocessing.converth5totiff as h5_converter
    import src.preprocessing.convertshapelabelstoraster as shapefile_converter
    import src.preprocessing.split_pixel as pixel_splitter
    import src.preprocessing.color_map as color_mapper
    print("   All preprocessing modules imported successfully")
    
    print("✅ Testing HDF5 converter function...")
    # Test if the function exists
    if hasattr(h5_converter, 'convert_h5_to_stacked_tiff'):
        print("   HDF5 converter function found")
    else:
        print("   ❌ HDF5 converter function not found")
    
    print("✅ Testing shapefile converter function...")
    # Test if the function exists
    if hasattr(shapefile_converter, 'convert_shapefile_to_raster'):
        print("   Shapefile converter function found")
    else:
        print("   ❌ Shapefile converter function not found")
    
    print("✅ Testing train/test mask function...")
    # Test if the function exists
    if hasattr(shapefile_converter, 'create_train_test_masks'):
        print("   Train/test mask function found")
    else:
        print("   ❌ Train/test mask function not found")
    
    print("\n🎉 All imports successful! The Streamlit app should work correctly.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)
