import h5py
import os
import numpy as np
import rasterio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.config import config
from rasterio.transform import from_bounds


def remove_bands(cube, bands_to_remove):
    """Remove specified band indices from a cube shaped (H, W, B)."""
    if not bands_to_remove:
        return cube
    all_bands = list(range(cube.shape[2]))
    keep_indices = [i for i in all_bands if i not in bands_to_remove]
    return cube[:, :, keep_indices]


def compute_transform_from_latlon(lat, lon, width, height):
    """
    Compute transform from lat/lon grid using bounding box.
    Assumes lat/lon are 2D arrays aligned to the data.
    """
    min_lon = np.min(lon)
    max_lon = np.max(lon)
    min_lat = np.min(lat)
    max_lat = np.max(lat)

    # GeoTransform from bounding box
    transform = from_bounds(min_lon, min_lat, max_lon, max_lat, width, height)
    return transform, "EPSG:4326"


def convert_h5_to_stacked_tiff(h5_path, vnir_key, swir_key, lat_key, lon_key, output_tiff_path,
                                vnir_bands_to_remove, swir_bands_to_remove):
    """
    Convert PRISMA HDF5 file to stacked GeoTIFF format.
    
    Args:
        h5_path (str): Path to input HDF5 file
        vnir_key (str): HDF5 key for VNIR data
        swir_key (str): HDF5 key for SWIR data
        lat_key (str): HDF5 key for latitude data
        lon_key (str): HDF5 key for longitude data
        output_tiff_path (str): Path for output TIFF file
        vnir_bands_to_remove (list): List of VNIR band indices to remove
        swir_bands_to_remove (list): List of SWIR band indices to remove
    
    Returns:
        str: Path to the created TIFF file
    """
    with h5py.File(h5_path, 'r') as h5_file:
        vnir_cube = np.array(h5_file[vnir_key], dtype=np.float32)
        swir_cube = np.array(h5_file[swir_key], dtype=np.float32)
        lat = np.array(h5_file[lat_key], dtype=np.float32)
        lon = np.array(h5_file[lon_key], dtype=np.float32)

    print(f"Original VNIR shape: {vnir_cube.shape}")
    print(f"Original SWIR shape: {swir_cube.shape}")

    # Transpose to (height, width, bands)
    if vnir_cube.shape[1] == 66:
        vnir_cube = np.transpose(vnir_cube, (0, 2, 1))
    if swir_cube.shape[1] == 173:
        swir_cube = np.transpose(swir_cube, (0, 2, 1))

    print(f"VNIR after transpose: {vnir_cube.shape}")
    print(f"SWIR after transpose: {swir_cube.shape}")

    vnir_cube = remove_bands(vnir_cube, vnir_bands_to_remove)
    swir_cube = remove_bands(swir_cube, swir_bands_to_remove)

    print(f"VNIR after band removal: {vnir_cube.shape}")
    print(f"SWIR after band removal: {swir_cube.shape}")

    # Validate shapes
    if vnir_cube.shape[:2] != swir_cube.shape[:2]:
        raise ValueError("VNIR and SWIR cubes must have the same spatial dimensions")
    if vnir_cube.shape[:2] != lat.shape or lat.shape != lon.shape:
        raise ValueError("Latitude/Longitude shapes must match VNIR/SWIR spatial size")

    # Stack and prepare output
    stacked_cube = np.concatenate((vnir_cube, swir_cube), axis=2)
    stacked_cube = np.transpose(stacked_cube, (2, 0, 1))  # (bands, height, width)

    height, width = stacked_cube.shape[1:]

    # Compute transform from bounds
    transform, crs = compute_transform_from_latlon(lat, lon, width, height)

    # Create nodata mask only for actual background
    mask = np.all(stacked_cube == 0.0, axis=0)
    nodata_value = -9999.0
    for band in range(stacked_cube.shape[0]):
        stacked_cube[band][mask] = nodata_value

    # Write GeoTIFF
    with rasterio.open(
        output_tiff_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=stacked_cube.shape[0],
        dtype='float32',
        transform=transform,
        crs=crs,
        nodata=nodata_value
    ) as dst:
        dst.write(stacked_cube)

    print(f"✅ GeoTIFF saved and placed correctly: {output_tiff_path}")
    return output_tiff_path


# === Main execution (only when run directly) ===
if __name__ == "__main__":
    convert_h5_to_stacked_tiff(
        h5_path=config.scene_path,
        vnir_key=config.h5_vnir_key,
        swir_key=config.h5_swir_key,
        lat_key=config.h5_lat_key,
        lon_key=config.h5_lon_key,
        output_tiff_path=config.tifpath,
        vnir_bands_to_remove=config.vnir_bands_to_remove,
        swir_bands_to_remove=config.swir_bands_to_remove
    )
