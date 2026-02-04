"""
Model inference module for object detection.
"""
import os
from typing import Dict, List, Optional
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import rasterio
from shapely.geometry import box
import geopandas as gpd
from tqdm import tqdm


class ObjectDetectionModel:
    """Object detection model wrapper."""

    def __init__(self, checkpoint_path: str, num_classes: int = 2, device: str = "cuda"):
        """
        Initialize model.

        Args:
            checkpoint_path: Path to model checkpoint (.pth)
            num_classes: Number of classes (including background)
            device: 'cuda' or 'cpu'
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.num_classes = num_classes
        self.model = self._load_model(checkpoint_path)
        print(f"Model loaded on {self.device}")

    def _get_model(self, num_classes: int):
        """Create Faster R-CNN model."""
        model = fasterrcnn_resnet50_fpn(pretrained=False)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    def _load_model(self, checkpoint_path: str):
        """Load model from checkpoint."""
        model = self._get_model(self.num_classes)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model

    @torch.no_grad()
    def predict_on_tile(self, image_tile: np.ndarray, score_threshold: float = 0.5) -> Dict:
        """
        Run inference on a single tile.

        Args:
            image_tile: RGB image array (H, W, 3)
            score_threshold: Minimum confidence score

        Returns:
            Dict with 'boxes', 'scores', 'labels'
        """
        # Convert to tensor
        image_tensor = torch.from_numpy(image_tile).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.to(self.device)

        # Predict
        predictions = self.model([image_tensor])[0]

        # Filter by score
        keep = predictions['scores'] > score_threshold

        return {
            'boxes': predictions['boxes'][keep].cpu().numpy(),
            'scores': predictions['scores'][keep].cpu().numpy(),
            'labels': predictions['labels'][keep].cpu().numpy()
        }

    def predict_on_geotiff(
        self,
        tif_path: str,
        tile_size: int = 1024,
        overlap: int = 128,
        score_threshold: float = 0.5,
        progress_callback: Optional[callable] = None
    ) -> gpd.GeoDataFrame:
        """
        Run inference on entire GeoTIFF.

        Args:
            tif_path: Path to GeoTIFF file
            tile_size: Size of tiles for inference
            overlap: Overlap between tiles
            score_threshold: Minimum confidence score
            progress_callback: Optional callback function(current, total)

        Returns:
            GeoDataFrame with detections
        """
        # Load image
        with rasterio.open(tif_path) as src:
            image_array = src.read([1, 2, 3]).transpose(1, 2, 0)
            transform = src.transform
            crs = src.crs

        h, w = image_array.shape[:2]
        step = tile_size - overlap

        # Calculate total tiles
        total_tiles = len(range(0, h, step)) * len(range(0, w, step))
        current_tile = 0

        all_detections = []

        # Process tiles
        for y in range(0, h, step):
            for x in range(0, w, step):
                # Extract tile
                tile = image_array[y:min(y+tile_size, h), x:min(x+tile_size, w)]

                # Predict
                predictions = self.predict_on_tile(tile, score_threshold)

                # Convert boxes to georeferenced coordinates
                for pred_box, score in zip(predictions['boxes'], predictions['scores']):
                    xmin, ymin, xmax, ymax = pred_box

                    # Convert to image coordinates
                    xmin_img = x + xmin
                    ymin_img = y + ymin
                    xmax_img = x + xmax
                    ymax_img = y + ymax

                    # Convert to georeferenced coordinates
                    minx_geo, maxy_geo = transform * (xmin_img, ymin_img)
                    maxx_geo, miny_geo = transform * (xmax_img, ymax_img)

                    # Create box geometry
                    bbox_geom = box(minx_geo, miny_geo, maxx_geo, maxy_geo)

                    all_detections.append({
                        'geometry': bbox_geom,
                        'score': float(score),
                        'class': 'object'
                    })

                # Update progress
                current_tile += 1
                if progress_callback:
                    progress_callback(current_tile, total_tiles)

        # Create GeoDataFrame
        if all_detections:
            gdf = gpd.GeoDataFrame(all_detections, crs=crs)
            gdf = gdf.sort_values('score', ascending=False).reset_index(drop=True)
            return gdf
        else:
            return None
