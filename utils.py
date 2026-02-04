"""
Utility functions for TIF processing and GeoJSON handling.
"""
import os
import io
import json
from typing import Tuple
import numpy as np
from PIL import Image
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas as gpd
from shapely.ops import unary_union


def generate_preview_image(tif_path: str, max_size: int = 1024) -> Tuple[bytes, dict]:
    """
    Generate a downsampled PNG preview of GeoTIFF for web display.

    Args:
        tif_path: Path to GeoTIFF file
        max_size: Maximum dimension for preview

    Returns:
        Tuple of (PNG bytes, metadata dict)
    """
    with rasterio.open(tif_path) as src:
        # Read RGB bands
        data = src.read([1, 2, 3])

        # Get metadata
        metadata = {
            'width': src.width,
            'height': src.height,
            'crs': str(src.crs),
            'bounds': list(src.bounds)
        }

        # Calculate resize factor
        h, w = data.shape[1], data.shape[2]
        scale = max_size / max(h, w)

        if scale < 1:
            new_h = int(h * scale)
            new_w = int(w * scale)
        else:
            new_h, new_w = h, w

        # Convert to PIL Image and resize
        img_array = np.moveaxis(data, 0, 2)  # CHW -> HWC
        img = Image.fromarray(img_array.astype(np.uint8))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Convert to PNG bytes
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_bytes = img_buffer.getvalue()

    return img_bytes, metadata


def generate_overlay_image(tif_path: str, max_size: int = 8192, format: str = 'JPEG', quality: int = 85) -> bytes:
    """
    Generate a high-resolution overlay image from GeoTIFF for map display.
    Reprojects to WGS84 (EPSG:4326) to match Leaflet's coordinate system.

    Args:
        tif_path: Path to GeoTIFF file
        max_size: Maximum dimension for overlay (default 8192px)
        format: Output format ('JPEG' or 'PNG')
        quality: JPEG quality (1-100, only used for JPEG)

    Returns:
        Image bytes in specified format
    """
    with rasterio.open(tif_path) as src:
        # Calculate transform to WGS84
        dst_crs = 'EPSG:4326'
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )

        # Apply max_size constraint
        scale = max_size / max(height, width)
        if scale < 1:
            width = int(width * scale)
            height = int(height * scale)
            # Adjust transform for new dimensions
            transform = transform * transform.scale(1/scale, 1/scale)

        # Create destination array for reprojected data
        dst_data = np.zeros((3, height, width), dtype=np.uint8)

        # Reproject each RGB band
        for i in range(3):
            reproject(
                source=rasterio.band(src, i + 1),
                destination=dst_data[i],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.lanczos
            )

        # Convert to PIL Image
        img_array = np.moveaxis(dst_data, 0, 2)  # CHW -> HWC
        img = Image.fromarray(img_array.astype(np.uint8))

        # Convert to bytes
        img_buffer = io.BytesIO()
        if format.upper() == 'JPEG':
            img.save(img_buffer, format='JPEG', quality=quality, optimize=True)
        else:
            img.save(img_buffer, format='PNG', optimize=True)

        img_bytes = img_buffer.getvalue()

    return img_bytes


def get_tif_bounds_geojson(tif_path: str) -> dict:
    """
    Get GeoTIFF bounds as GeoJSON for map initialization.

    Args:
        tif_path: Path to GeoTIFF file

    Returns:
        GeoJSON feature collection with bounds polygon
    """
    with rasterio.open(tif_path) as src:
        bounds = src.bounds
        crs = src.crs

        # Create bounds polygon
        from shapely.geometry import box
        bounds_poly = box(bounds.left, bounds.bottom, bounds.right, bounds.top)

        # Create GeoDataFrame and convert to WGS84 for Leaflet
        gdf = gpd.GeoDataFrame([{'geometry': bounds_poly}], crs=crs)
        gdf_wgs84 = gdf.to_crs('EPSG:4326')

        # Convert to GeoJSON
        return json.loads(gdf_wgs84.to_json())


def merge_overlapping_geometries(gdf: gpd.GeoDataFrame, buffer_dist: float = 0.1) -> gpd.GeoDataFrame:
    """
    Merge overlapping or nearby geometries in GeoDataFrame.

    Args:
        gdf: GeoDataFrame with detections
        buffer_dist: Distance for buffering/merging (in CRS units)

    Returns:
        GeoDataFrame with merged geometries
    """
    if gdf is None or len(gdf) == 0:
        return gdf

    # Buffer slightly to merge nearby boxes
    buffered = gdf.geometry.buffer(buffer_dist)

    # Union all overlapping geometries
    merged = unary_union(buffered)

    # Unbuffer to get original size back
    if hasattr(merged, 'geoms'):
        # MultiPolygon
        final_geoms = [geom.buffer(-buffer_dist) for geom in merged.geoms]
    else:
        # Single Polygon
        final_geoms = [merged.buffer(-buffer_dist)]

    # Create new GeoDataFrame
    merged_gdf = gpd.GeoDataFrame(
        {'geometry': final_geoms, 'class': 'object'},
        crs=gdf.crs
    )

    return merged_gdf


def gdf_to_geojson_for_leaflet(gdf: gpd.GeoDataFrame) -> dict:
    """
    Convert GeoDataFrame to GeoJSON in WGS84 for Leaflet display.

    Args:
        gdf: GeoDataFrame in any CRS

    Returns:
        GeoJSON dict in EPSG:4326
    """
    if gdf is None or len(gdf) == 0:
        return {"type": "FeatureCollection", "features": []}

    # Convert to WGS84
    gdf_wgs84 = gdf.to_crs('EPSG:4326')

    # Convert to GeoJSON
    return json.loads(gdf_wgs84.to_json())


def cleanup_old_files(directory: str, max_age_hours: int = 1):
    """
    Clean up old uploaded files.

    Args:
        directory: Directory to clean
        max_age_hours: Maximum age in hours
    """
    import time

    if not os.path.exists(directory):
        return

    current_time = time.time()
    max_age_seconds = max_age_hours * 3600

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            file_age = current_time - os.path.getmtime(filepath)
            if file_age > max_age_seconds:
                try:
                    os.remove(filepath)
                    print(f"Cleaned up old file: {filename}")
                except Exception as e:
                    print(f"Error removing {filename}: {e}")
