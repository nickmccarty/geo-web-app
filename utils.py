"""
Utility functions for TIF processing and GeoJSON handling.
"""
import os
import io
import json
from typing import Tuple
import numpy as np
import pandas as pd
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


def load_qc_points(csv_bytes: bytes) -> pd.DataFrame:
    """
    Parse QC check points CSV. Supports both header-based and positional column mapping.

    Header-based: recognizes columns named y, x, point_id, z (any order).
    Positional fallback: assumes column order point_id, y, x, z.

    Args:
        csv_bytes: Raw CSV file bytes

    Returns:
        DataFrame with columns: point_id, y, x, z
    """
    # First pass: check if first row is a header
    raw = pd.read_csv(io.BytesIO(csv_bytes), header=None, dtype=str)

    if len(raw.columns) < 2:
        raise ValueError(f"CSV must have at least 2 columns (y, x), found {len(raw.columns)}")

    # Check if first row contains recognizable header names
    first_row_lower = [str(v).strip().lower() for v in raw.iloc[0]]
    known_headers = {'y', 'x', 'z', 'point_id', 'northing', 'easting', 'elev', 'id'}
    has_header = any(val in known_headers for val in first_row_lower)

    if has_header:
        # Re-read with header
        df = pd.read_csv(io.BytesIO(csv_bytes), dtype=str)
        df.columns = [c.strip().lower() for c in df.columns]

        # Map common aliases
        col_map = {}
        for c in df.columns:
            if c in ('y', 'northing'):
                col_map['y'] = c
            elif c in ('x', 'easting'):
                col_map['x'] = c
            elif c in ('point_id', 'id'):
                col_map['point_id'] = c
            elif c in ('z', 'elev', 'elevation'):
                col_map['z'] = c

        if 'y' not in col_map or 'x' not in col_map:
            raise ValueError(
                f"CSV header must include 'y' and 'x' columns. Found: {list(df.columns)}"
            )

        out = pd.DataFrame({
            'y': pd.to_numeric(df[col_map['y']], errors='coerce'),
            'x': pd.to_numeric(df[col_map['x']], errors='coerce'),
            'point_id': pd.to_numeric(
                df[col_map['point_id']].astype(str).str.replace(r'^\s*CK\s*', '', regex=True).str.strip(),
                errors='coerce'
            ) if 'point_id' in col_map else None,
            'z': pd.to_numeric(df[col_map['z']], errors='coerce') if 'z' in col_map else 0,
        }).reset_index(drop=True)
    else:
        # Positional: point_id, y, x, z
        if len(raw.columns) < 3:
            raise ValueError(f"CSV without header must have at least 3 columns (point_id, y, x)")

        raw[0] = raw[0].astype(str).str.replace(r'^\s*CK\s*', '', regex=True).str.strip()

        out = pd.DataFrame({
            'point_id': pd.to_numeric(raw[0], errors='coerce'),
            'y': pd.to_numeric(raw[1], errors='coerce'),
            'x': pd.to_numeric(raw[2], errors='coerce'),
            'z': pd.to_numeric(raw[3], errors='coerce') if len(raw.columns) >= 4 else 0,
        }).reset_index(drop=True)

    if out['point_id'].isnull().all() or 'point_id' not in out.columns or out['point_id'] is None:
        out['point_id'] = out.index + 1

    return out


def compute_deviations(csv_bytes: bytes, geojson_path: str, tif_path: str) -> dict:
    """
    Compute QC point deviations against detection polygons.

    Loads QC points (in TIF's native CRS), finds containing detection polygons,
    computes distance from each point to the polygon centroid (converted to cm),
    and reprojects results to WGS84 for map display.

    Args:
        csv_bytes: Raw CSV file bytes for QC points
        geojson_path: Path to detections GeoJSON (native CRS)
        tif_path: Path to source GeoTIFF (for CRS and bounds)

    Returns:
        Dict with qc_points list and summary stats
    """
    THRESHOLD_CM = 3.0
    FEET_TO_CM = 30.48

    # Load QC points
    qc_df = load_qc_points(csv_bytes)

    # Get TIF CRS and bounds
    with rasterio.open(tif_path) as src:
        tif_crs = src.crs
        tif_bounds = src.bounds

    # Load detection polygons (in native CRS)
    polygons_gdf = gpd.read_file(geojson_path)
    if polygons_gdf.crs is None:
        polygons_gdf = polygons_gdf.set_crs(tif_crs)
    elif str(polygons_gdf.crs) != str(tif_crs):
        polygons_gdf = polygons_gdf.to_crs(tif_crs)

    # Drop rows with missing coordinates
    qc_df = qc_df.dropna(subset=['x', 'y']).reset_index(drop=True)

    if len(qc_df) == 0:
        # Re-read raw CSV to show what was parsed
        raw_df = pd.read_csv(io.BytesIO(csv_bytes), header=None, dtype=str, nrows=3)
        preview = raw_df.to_string(index=False)
        raise ValueError(
            f"No valid QC points after parsing. Check column order (expected: point_id, y, x, z). "
            f"First rows of CSV:\n{preview}"
        )

    # Convert QC points to GeoDataFrame in TIF CRS
    points_gdf = gpd.GeoDataFrame(
        qc_df,
        geometry=gpd.points_from_xy(qc_df['x'], qc_df['y']),
        crs=tif_crs
    )

    # Filter to TIF bounds
    left, bottom, right, top = tif_bounds
    in_bounds = (
        (points_gdf.geometry.x >= left) &
        (points_gdf.geometry.x <= right) &
        (points_gdf.geometry.y >= bottom) &
        (points_gdf.geometry.y <= top)
    )
    points_filtered = points_gdf[in_bounds].copy()

    # Compute deviations
    results = []
    for _, point in points_filtered.iterrows():
        containing = polygons_gdf[polygons_gdf.contains(point.geometry)]

        if not containing.empty:
            # Find the closest centroid among containing polygons
            best_dist = float('inf')
            best_poly_id = None
            best_centroid = None

            for poly_idx, poly in containing.iterrows():
                centroid = poly.geometry.centroid
                dist = point.geometry.distance(centroid)
                if dist < best_dist:
                    best_dist = dist
                    best_poly_id = poly_idx
                    best_centroid = centroid

            deviation_cm = best_dist * FEET_TO_CM

            results.append({
                'point_id': int(point['point_id']) if pd.notna(point['point_id']) else 0,
                'matched': True,
                'deviation_cm': round(deviation_cm, 2),
                'exceeds_threshold': deviation_cm > THRESHOLD_CM,
                'point_geom': point.geometry,
                'centroid_geom': best_centroid,
            })
        else:
            results.append({
                'point_id': int(point['point_id']) if pd.notna(point['point_id']) else 0,
                'matched': False,
                'deviation_cm': None,
                'exceeds_threshold': False,
                'point_geom': point.geometry,
                'centroid_geom': None,
            })

    # Batch reproject QC points to WGS84
    pts_gdf = gpd.GeoDataFrame(
        [{'idx': i, 'geometry': r['point_geom']} for i, r in enumerate(results)],
        crs=tif_crs
    ).to_crs('EPSG:4326')

    # Batch reproject matched centroids to WGS84
    centroid_rows = [
        {'idx': i, 'geometry': r['centroid_geom']}
        for i, r in enumerate(results) if r['centroid_geom'] is not None
    ]
    if centroid_rows:
        ct_gdf = gpd.GeoDataFrame(centroid_rows, crs=tif_crs).to_crs('EPSG:4326')
        centroid_map = {int(row['idx']): row.geometry for _, row in ct_gdf.iterrows()}
    else:
        centroid_map = {}

    # Build output
    qc_points_out = []
    for i, r in enumerate(results):
        pt_wgs = pts_gdf.loc[pts_gdf['idx'] == i].geometry.iloc[0]
        entry = {
            'point_id': r['point_id'],
            'lat': round(pt_wgs.y, 8),
            'lng': round(pt_wgs.x, 8),
            'matched': r['matched'],
            'deviation_cm': r['deviation_cm'],
            'exceeds_threshold': r['exceeds_threshold'],
            'centroid_lat': None,
            'centroid_lng': None,
        }

        if i in centroid_map:
            ct_wgs = centroid_map[i]
            entry['centroid_lat'] = round(ct_wgs.y, 8)
            entry['centroid_lng'] = round(ct_wgs.x, 8)

        qc_points_out.append(entry)

    # Summary stats
    matched = [r for r in results if r['matched']]
    deviations = [r['deviation_cm'] for r in matched if r['deviation_cm'] is not None]

    summary = {
        'total_qc_points': len(points_filtered),
        'matched_points': len(matched),
        'unmatched_points': len(points_filtered) - len(matched),
        'avg_deviation_cm': round(sum(deviations) / len(deviations), 2) if deviations else 0,
        'max_deviation_cm': round(max(deviations), 2) if deviations else 0,
        'count_exceeding_3cm': sum(1 for d in deviations if d > THRESHOLD_CM),
        'threshold_cm': THRESHOLD_CM,
    }

    return {
        'qc_points': qc_points_out,
        'summary': summary,
    }


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
