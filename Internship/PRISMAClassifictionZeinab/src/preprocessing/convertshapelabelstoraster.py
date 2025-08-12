import geopandas as gpd
import rasterio
from rasterio import features
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.config import config


def convert_shapefile_to_raster(shapefile_path, reference_tif_path, output_label_tif_path, class_field="Classcode"):
    """
    Convert shapefile with class labels to raster format.
    
    Args:
        shapefile_path (str): Path to input shapefile
        reference_tif_path (str): Path to reference raster for georeference
        output_label_tif_path (str): Path for output label raster
        class_field (str): Name of the field containing class labels
    
    Returns:
        str: Path to the created raster file
    """
    # === Load shapefile ===
    gdf = gpd.read_file(shapefile_path)
    
    # === Class field name (adjust if needed) ===
    gdf[class_field] = gdf[class_field].astype(int)
    
    # === Shift class values by +2 (1 → 3, 2 → 4, ..., 10 → 12) ===
    gdf[class_field] = gdf[class_field] + 2
    
    # === Open reference raster ===
    with rasterio.open(reference_tif_path) as src:
        ref_crs = src.crs
        ref_transform = src.transform
        ref_shape = (src.height, src.width)
        ref_profile = src.profile
        data_mask = ~src.read(1, masked=True).mask  # True where there is image data
    
    # === Reproject shapefile to match raster CRS if needed ===
    if gdf.crs != ref_crs:
        gdf = gdf.to_crs(ref_crs)
    
    # === Initialize raster with UNCLASSIFIED = 2 ===
    burned = np.full(ref_shape, 2, dtype=np.uint8)
    
    # === Rasterize shapefile into array ===
    shapes = ((geom, val) for geom, val in zip(gdf.geometry, gdf[class_field]))
    
    burned_from_shapes = features.rasterize(
        shapes=shapes,
        out_shape=ref_shape,
        transform=ref_transform,
        fill=2,         # Unclassified
        dtype=np.uint8
    )
    
    # === Insert burned shapes into array ===
    burned[:, :] = burned_from_shapes
    
    # === Set BACKGROUND = 1 (outside image data area) ===
    burned[~data_mask] = 1
    
    # === Update profile and save raster ===
    ref_profile.update({
        "count": 1,
        "dtype": "uint8",
        "nodata": 0,
        "compress": "lzw"
    })
    
    with rasterio.open(output_label_tif_path, "w", **ref_profile) as dst:
        dst.write(burned, 1)
    
    print("✅ Raster saved with background=1, unclassified=2, and shifted classes from 3+")
    return output_label_tif_path


def create_train_test_masks(label_raster_path, train_mask_path, test_mask_path, test_ratio=0.2, samples_per_class=1000):
    """
    Create training and testing masks from a label raster.
    
    Args:
        label_raster_path (str): Path to the label raster
        train_mask_path (str): Path for training mask output
        test_mask_path (str): Path for testing mask output
        test_ratio (float): Ratio of data to use for testing (0.0 to 1.0)
        samples_per_class (int): Number of samples to take per class
    
    Returns:
        tuple: Paths to training and testing masks
    """
    from sklearn.model_selection import train_test_split
    
    # === Load labeled image ===
    with rasterio.open(label_raster_path) as src:
        labels = src.read(1)
        profile = src.profile
    
    # === Identify unique classes (excluding background = 0) ===
    unique_classes = np.unique(labels)
    unique_classes = unique_classes[unique_classes > 0]
    
    # === Create empty train and test masks ===
    train_mask = np.zeros_like(labels, dtype=np.uint8)
    test_mask = np.zeros_like(labels, dtype=np.uint8)
    
    # === For each class, randomly sample pixels and split into train/test ===
    np.random.seed(42)
    
    for cls in unique_classes:
        # Get all pixel coordinates for this class
        y_cls, x_cls = np.where(labels == cls)
        coords_cls = list(zip(y_cls, x_cls))
        
        # Shuffle and take samples
        selected_coords = np.array(coords_cls)
        np.random.shuffle(selected_coords)
        selected_coords = selected_coords[:min(samples_per_class, len(selected_coords))]
        
        # Split into train/test
        coords_train, coords_test = train_test_split(
            selected_coords, test_size=test_ratio, random_state=42
        )
        
        # Fill train mask
        for y, x in coords_train:
            train_mask[y, x] = cls
        
        # Fill test mask
        for y, x in coords_test:
            test_mask[y, x] = cls
    
    # === Save the masks ===
    profile.update(dtype='uint8', count=1, compress='lzw')
    
    with rasterio.open(train_mask_path, "w", **profile) as dst:
        dst.write(train_mask, 1)
    
    with rasterio.open(test_mask_path, "w", **profile) as dst:
        dst.write(test_mask, 1)
    
    print(f"✅ Train and test masks saved with {samples_per_class} samples per class across {len(unique_classes)} classes.")
    return train_mask_path, test_mask_path


# === Main execution (only when run directly) ===
if __name__ == "__main__":
    # === INPUT PATHS ===
    reference_tif = config.tifpath                       # Your image raster
    shapefile_path = config.shapefile_path               # Your shapefile
    output_label_tif = config.new_final_class_labels     # Output label raster
    
    # Convert shapefile to raster
    convert_shapefile_to_raster(
        shapefile_path=shapefile_path,
        reference_tif_path=reference_tif,
        output_label_tif_path=output_label_tif,
        class_field=config.class_field
    )
    
    # Create train/test masks
    create_train_test_masks(
        label_raster_path=output_label_tif,
        train_mask_path=config.train_mask_path,
        test_mask_path=config.test_mask_path
    )
