#!/usr/bin/env python3
"""
Test script to verify folium and mapping functionality
"""

import sys
import os

print("Testing folium and mapping functionality...")

try:
    print("✅ Importing folium...")
    import folium
    print(f"   Folium version: {folium.__version__}")
    
    print("✅ Importing folium plugins...")
    from folium import plugins
    print("   Plugins imported successfully")
    
    print("✅ Testing basic map creation...")
    m = folium.Map(location=[30.0, 31.0], zoom_start=10)
    print("   Basic map created successfully")
    
    print("✅ Testing tile layer addition...")
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google Satellite',
        name='Google Satellite'
    ).add_to(m)
    print("   Google Satellite tiles added successfully")
    
    print("✅ Testing map save...")
    test_map_path = "test_map.html"
    m.save(test_map_path)
    print(f"   Test map saved to: {test_map_path}")
    
    # Clean up
    if os.path.exists(test_map_path):
        os.remove(test_map_path)
        print("   Test map cleaned up")
    
    print("\n🎉 All folium functionality working correctly!")
    print("The Google Satellite visualization should work in your Streamlit app.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install folium: pip install folium")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)
