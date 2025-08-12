from logging import config
import numpy as np
import rasterio
from rasterio.enums import ColorInterp
from rasterio.io import MemoryFile
from rasterio.plot import show
from matplotlib.colors import to_rgba
import  os
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.config import config
 

# === Load your classified label image

input_tif = config.new_final_class_labels
output_colored_tif = config.label_tif_path

with rasterio.open(input_tif) as src:
    label_data = src.read(1)
    profile = src.profile.copy()

# === Define class colormap (value: (R, G, B)) from class 0 to class 10+
custom_cmap = {
    0: (0, 0, 0),         # Class 0 = black (background/nodata)
    1: (127, 127, 127),   # Class 1 = gray (outside bounds)
    2: (255, 255, 255),   # Class 2 = white (unclassified)
    3: (255, 0, 0),       # Class 3 = red
    4: (0, 255, 0),       # Class 4 = green
    5: (0, 0, 255),       # Class 5 = blue
    6: (255, 255, 0),     # Class 6 = yellow
    7: (255, 0, 255),     # Class 7 = magenta
    8: (0, 255, 255),     # Class 8 = cyan
    9: (160, 32, 240),    # Class 9 = purple
    10: (255, 165, 0),    # Class 10 = orange
    11: (128, 0, 0),      # Class 11 = maroon
    12: (0, 128, 128),    # Class 12 = teal
}

# === Update profile
profile.update({
    'count': 1,
    'dtype': 'uint8',
    'compress': 'lzw'
})

# === Save the TIFF with color map
with rasterio.open(output_colored_tif, 'w', **profile) as dst:
    dst.write(label_data, 1)
    dst.write_colormap(1, custom_cmap)

print(f"✅ Colorized TIFF saved to: {output_colored_tif}")
