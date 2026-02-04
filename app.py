"""
FastAPI application for GeoTIFF object detection.
"""
import os
import uuid
import asyncio
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import aiofiles

from model_inference import ObjectDetectionModel
from utils import (
    generate_preview_image,
    generate_overlay_image,
    get_tif_bounds_geojson,
    merge_overlapping_geometries,
    gdf_to_geojson_for_leaflet,
    cleanup_old_files
)

# Configuration
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "best_model.pth"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(title="GeoTIFF Object Detection", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Templates
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Global model instance
model: Optional[ObjectDetectionModel] = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global model

    if not CHECKPOINT_PATH.exists():
        print(f"⚠️  Warning: Model checkpoint not found at {CHECKPOINT_PATH}")
        print("Please copy your best_model.pth to the checkpoints/ directory")
    else:
        print("Loading model...")
        model = ObjectDetectionModel(
            checkpoint_path=str(CHECKPOINT_PATH),
            num_classes=2,
            device="cuda"
        )
        print("✓ Model loaded successfully")

    # Cleanup old files
    cleanup_old_files(str(UPLOAD_DIR), max_age_hours=1)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a GeoTIFF file.

    Returns:
        JSON with file_id, preview image, and metadata
    """
    # Validate file extension
    if not (file.filename.lower().endswith('.tif') or file.filename.lower().endswith('.tiff')):
        raise HTTPException(status_code=400, detail="Only .tif and .tiff files are supported")

    # Generate unique file ID
    file_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{file_id}.tif"

    # Save uploaded file
    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)

    try:
        # Generate preview image
        preview_bytes, metadata = generate_preview_image(str(file_path))

        # Save preview
        preview_path = UPLOAD_DIR / f"{file_id}_preview.png"
        async with aiofiles.open(preview_path, 'wb') as f:
            await f.write(preview_bytes)

        # Generate high-resolution overlay image
        overlay_bytes = generate_overlay_image(str(file_path), max_size=8192, format='JPEG', quality=90)

        # Save overlay
        overlay_path = UPLOAD_DIR / f"{file_id}_overlay.jpg"
        async with aiofiles.open(overlay_path, 'wb') as f:
            await f.write(overlay_bytes)

        # Get bounds for map
        bounds_geojson = get_tif_bounds_geojson(str(file_path))

        return JSONResponse({
            "file_id": file_id,
            "filename": file.filename,
            "preview_url": f"/static/uploads/{file_id}_preview.png",
            "overlay_url": f"/static/uploads/{file_id}_overlay.jpg",
            "metadata": metadata,
            "bounds": bounds_geojson
        })

    except Exception as e:
        # Cleanup on error
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.websocket("/ws/inference/{file_id}")
async def websocket_inference(websocket: WebSocket, file_id: str):
    """
    WebSocket endpoint for running inference with progress updates.

    Args:
        file_id: Unique file identifier
    """
    await websocket.accept()

    if model is None:
        await websocket.send_json({
            "type": "error",
            "message": "Model not loaded. Please check server logs."
        })
        await websocket.close()
        return

    file_path = UPLOAD_DIR / f"{file_id}.tif"

    if not file_path.exists():
        await websocket.send_json({
            "type": "error",
            "message": "File not found"
        })
        await websocket.close()
        return

    try:
        # Progress callback
        async def progress_callback(current: int, total: int):
            await websocket.send_json({
                "type": "progress",
                "current": current,
                "total": total,
                "percentage": int((current / total) * 100)
            })

        # Send starting message
        await websocket.send_json({"type": "status", "message": "Starting inference..."})

        # Run inference in executor to avoid blocking
        loop = asyncio.get_event_loop()

        def run_inference():
            def sync_progress(current, total):
                # Schedule async callback
                asyncio.run_coroutine_threadsafe(
                    progress_callback(current, total),
                    loop
                ).result()

            return model.predict_on_geotiff(
                str(file_path),
                tile_size=1024,
                overlap=128,
                score_threshold=0.5,
                progress_callback=sync_progress
            )

        # Run in thread pool
        detections_gdf = await loop.run_in_executor(None, run_inference)

        if detections_gdf is None or len(detections_gdf) == 0:
            await websocket.send_json({
                "type": "complete",
                "detections": {"type": "FeatureCollection", "features": []},
                "count": 0
            })
        else:
            # Merge overlapping geometries
            await websocket.send_json({"type": "status", "message": "Merging detections..."})
            merged_gdf = merge_overlapping_geometries(detections_gdf, buffer_dist=0.5)

            # Convert to GeoJSON for Leaflet
            geojson = gdf_to_geojson_for_leaflet(merged_gdf)

            # Save results
            results_path = UPLOAD_DIR / f"{file_id}_detections.geojson"
            merged_gdf.to_file(results_path, driver='GeoJSON')

            await websocket.send_json({
                "type": "complete",
                "detections": geojson,
                "count": len(merged_gdf),
                "download_url": f"/download/{file_id}"
            })

    except WebSocketDisconnect:
        print(f"WebSocket disconnected for {file_id}")
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": f"Inference error: {str(e)}"
        })
    finally:
        await websocket.close()


@app.get("/download/{file_id}")
async def download_results(file_id: str):
    """
    Download detection results as GeoJSON.

    Args:
        file_id: Unique file identifier
    """
    results_path = UPLOAD_DIR / f"{file_id}_detections.geojson"

    if not results_path.exists():
        raise HTTPException(status_code=404, detail="Results not found")

    async with aiofiles.open(results_path, 'rb') as f:
        content = await f.read()

    return Response(
        content=content,
        media_type="application/geo+json",
        headers={"Content-Disposition": f"attachment; filename=detections_{file_id}.geojson"}
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
