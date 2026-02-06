"""
FastAPI application for GeoTIFF object detection.
"""
import os
import uuid
import asyncio
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from starlette.websockets import WebSocketState
from fastapi.responses import HTMLResponse, JSONResponse, Response, FileResponse
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
    compute_deviations,
    generate_deviation_report,
    cleanup_old_files
)

# Configuration
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(title="GeoTIFF Object Detection", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Templates
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Global model instance
model: Optional[ObjectDetectionModel] = None
current_checkpoint_name: Optional[str] = None
current_device: str = "cuda"


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global model, current_checkpoint_name, current_device

    # Find default checkpoint (prefer best_model.pth, else first .pth found)
    default_path = CHECKPOINTS_DIR / "best_model.pth"
    if not default_path.exists():
        pth_files = sorted(CHECKPOINTS_DIR.glob("*.pth"))
        default_path = pth_files[0] if pth_files else None

    if default_path is None or not default_path.exists():
        print(f"Warning: No .pth checkpoints found in {CHECKPOINTS_DIR}")
    else:
        print(f"Loading model: {default_path.name}...")
        model = ObjectDetectionModel(
            checkpoint_path=str(default_path),
            num_classes=2,
            device="cuda"
        )
        current_checkpoint_name = default_path.name
        current_device = model.device
        print(f"Model loaded successfully on {current_device}")

    # Cleanup old files
    cleanup_old_files(str(UPLOAD_DIR), max_age_hours=1)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/favicon.ico")
async def favicon():
    """Serve the favicon to avoid 404."""
    return FileResponse("static/favicon.ico")


@app.get("/models")
async def list_models():
    """List available model checkpoints and current model state."""
    import torch
    pth_files = sorted([f.name for f in CHECKPOINTS_DIR.glob("*.pth")])
    return JSONResponse({
        "checkpoints": pth_files,
        "current_checkpoint": current_checkpoint_name,
        "current_device": current_device,
        "cuda_available": torch.cuda.is_available()
    })


@app.post("/models/load")
async def load_model(request: Request):
    """Load a model checkpoint on the specified device."""
    global model, current_checkpoint_name, current_device

    body = await request.json()
    checkpoint = body.get("checkpoint")
    device = body.get("device", "cuda")

    if not checkpoint:
        raise HTTPException(status_code=400, detail="checkpoint is required")

    checkpoint_path = CHECKPOINTS_DIR / checkpoint
    if not checkpoint_path.exists():
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {checkpoint}")

    if device not in ("cuda", "cpu"):
        raise HTTPException(status_code=400, detail="device must be 'cuda' or 'cpu'")

    try:
        loop = asyncio.get_event_loop()

        def do_load():
            return ObjectDetectionModel(
                checkpoint_path=str(checkpoint_path),
                num_classes=2,
                device=device
            )

        new_model = await loop.run_in_executor(None, do_load)
        model = new_model
        current_checkpoint_name = checkpoint
        current_device = model.device
        return JSONResponse({
            "status": "ok",
            "checkpoint": current_checkpoint_name,
            "device": current_device
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a GeoTIFF file (save only).

    Returns:
        JSON with file_id and filename. Heavy processing happens via /ws/process/{file_id}.
    """
    if not (file.filename.lower().endswith('.tif') or file.filename.lower().endswith('.tiff')):
        raise HTTPException(status_code=400, detail="Only .tif and .tiff files are supported")

    file_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{file_id}.tif"

    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)

    return JSONResponse({"file_id": file_id, "filename": file.filename})


# Store original filenames so the process WS can include them
_uploaded_filenames: dict = {}


@app.websocket("/ws/process/{file_id}")
async def websocket_process(websocket: WebSocket, file_id: str):
    """
    WebSocket endpoint for processing an uploaded GeoTIFF.
    Streams real log messages as preview, overlay, and bounds are generated.
    """
    await websocket.accept()

    file_path = UPLOAD_DIR / f"{file_id}.tif"
    if not file_path.exists():
        await websocket.send_json({"type": "error", "message": "File not found"})
        await websocket.close()
        return

    # Keep-alive ping
    async def keep_alive():
        try:
            while websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json({"type": "ping"})
                await asyncio.sleep(15)
        except Exception:
            pass

    ping_task = asyncio.create_task(keep_alive())
    loop = asyncio.get_event_loop()

    async def send_log(message: str):
        await websocket.send_json({"type": "log", "message": message})

    try:
        await send_log("Processing started.")

        # --- Preview ---
        await send_log("Generating preview image...")
        preview_bytes, metadata = await loop.run_in_executor(
            None, lambda: generate_preview_image(str(file_path), max_size=4096)
        )
        preview_path = UPLOAD_DIR / f"{file_id}_preview.png"
        async with aiofiles.open(preview_path, 'wb') as f:
            await f.write(preview_bytes)
        await send_log(f"Preview saved ({metadata['width']}x{metadata['height']} source).")

        # --- Overlay ---
        await send_log("Generating high-resolution overlay...")
        await send_log("Reprojecting for map display...")
        overlay_bytes = await loop.run_in_executor(
            None, lambda: generate_overlay_image(str(file_path), max_size=8192, format='JPEG', quality=90)
        )
        overlay_path = UPLOAD_DIR / f"{file_id}_overlay.jpg"
        async with aiofiles.open(overlay_path, 'wb') as f:
            await f.write(overlay_bytes)
        overlay_size_mb = len(overlay_bytes) / (1024 * 1024)
        await send_log(f"Overlay saved ({overlay_size_mb:.1f} MB).")

        # --- Bounds ---
        await send_log("Calculating geo bounds...")
        bounds_geojson = await loop.run_in_executor(
            None, lambda: get_tif_bounds_geojson(str(file_path))
        )
        await send_log("Bounds calculated. Ready.")

        # Send complete with all metadata
        await websocket.send_json({
            "type": "complete",
            "file_id": file_id,
            "preview_url": f"/static/uploads/{file_id}_preview.png",
            "overlay_url": f"/static/uploads/{file_id}_overlay.jpg",
            "metadata": metadata,
            "bounds": bounds_geojson
        })

    except WebSocketDisconnect:
        print(f"Process WS disconnected for {file_id}")
    except Exception as e:
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Processing error: {str(e)}"
            })
        except Exception:
            pass
        # Cleanup on error
        if file_path.exists():
            file_path.unlink()
    finally:
        ping_task.cancel()
        try:
            await websocket.close()
        except Exception:
            pass


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

    # Keep-alive ping to prevent WebSocket timeout on large files
    async def keep_alive():
        try:
            while websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json({"type": "ping"})
                await asyncio.sleep(15)
        except Exception:
            pass

    ping_task = asyncio.create_task(keep_alive())

    try:
        # Progress callback (async, called from sync thread via run_coroutine_threadsafe)
        async def progress_callback(current: int, total: int):
            await websocket.send_json({
                "type": "progress",
                "current": current,
                "total": total,
                "percentage": int((current / total) * 100)
            })

        # Send starting message
        await websocket.send_json({"type": "status", "message": "Starting inference..."})
        await websocket.send_json({"type": "log", "message": "Model loaded. Beginning tile-based inference..."})

        # Run inference in executor to avoid blocking the event loop.
        # The model calls progress_callback synchronously from the thread,
        # so we bridge it with run_coroutine_threadsafe.
        loop = asyncio.get_event_loop()

        def run_inference():
            def sync_progress(current, total):
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
            await websocket.send_json({"type": "log", "message": f"Raw detections: {len(detections_gdf)}. Merging overlaps (buffer=0.5)..."})
            merged_gdf = merge_overlapping_geometries(detections_gdf, buffer_dist=0.5)
            await websocket.send_json({"type": "log", "message": f"Merged to {len(merged_gdf)} detections."})

            # Convert to GeoJSON for Leaflet
            await websocket.send_json({"type": "log", "message": "Converting to GeoJSON..."})
            geojson = gdf_to_geojson_for_leaflet(merged_gdf)

            # Save results
            results_path = UPLOAD_DIR / f"{file_id}_detections.geojson"
            merged_gdf.to_file(results_path, driver='GeoJSON')
            await websocket.send_json({"type": "log", "message": "Results saved. Done."})

            await websocket.send_json({
                "type": "complete",
                "detections": geojson,
                "count": len(merged_gdf),
                "download_url": f"/download/{file_id}"
            })

    except WebSocketDisconnect:
        print(f"WebSocket disconnected for {file_id}")
    except Exception as e:
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Inference error: {str(e)}"
            })
        except Exception:
            pass
    finally:
        ping_task.cancel()
        try:
            await websocket.close()
        except Exception:
            pass


@app.post("/qc-analysis/{file_id}")
async def qc_analysis(file_id: str, file: UploadFile = File(...)):
    """
    Run QC deviation analysis against detection results.

    Accepts a QC check points CSV, computes deviations against
    the saved detection polygons, and returns results as JSON.
    """
    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only .csv files are supported")

    tif_path = UPLOAD_DIR / f"{file_id}.tif"
    if not tif_path.exists():
        raise HTTPException(status_code=404, detail="Source file not found")

    geojson_path = UPLOAD_DIR / f"{file_id}_detections.geojson"
    if not geojson_path.exists():
        raise HTTPException(status_code=404, detail="No detections found. Run inference first.")

    csv_bytes = await file.read()

    # Save CSV so the report endpoint can access it later
    csv_path = UPLOAD_DIR / f"{file_id}_qc.csv"
    async with aiofiles.open(csv_path, 'wb') as f:
        await f.write(csv_bytes)

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: compute_deviations(csv_bytes, str(geojson_path), str(tif_path))
        )
        return JSONResponse(result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"CSV parsing error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"QC analysis error: {str(e)}")


@app.websocket("/ws/qc-report/{file_id}")
async def ws_qc_report(websocket: WebSocket, file_id: str):
    """
    WebSocket endpoint for generating QC deviation report with progress logs.
    Saves the report to disk and sends a download URL on completion.
    """
    await websocket.accept()

    csv_path = UPLOAD_DIR / f"{file_id}_qc.csv"
    geojson_path = UPLOAD_DIR / f"{file_id}_detections.geojson"
    tif_path = UPLOAD_DIR / f"{file_id}.tif"

    for path, label in [(csv_path, "QC CSV"), (geojson_path, "Detections"), (tif_path, "Source TIF")]:
        if not path.exists():
            await websocket.send_json({"type": "error", "message": f"{label} not found."})
            await websocket.close()
            return

    loop = asyncio.get_event_loop()

    async def send_log(message: str):
        await websocket.send_json({"type": "log", "message": message})

    try:
        async with aiofiles.open(csv_path, 'rb') as f:
            csv_bytes = await f.read()

        def run_report():
            def sync_log(msg):
                asyncio.run_coroutine_threadsafe(send_log(msg), loop).result()
            return generate_deviation_report(
                csv_bytes, str(geojson_path), str(tif_path), log_callback=sync_log
            )

        xlsx_bytes = await loop.run_in_executor(None, run_report)

        # Save to disk so the GET endpoint can serve it
        report_path = UPLOAD_DIR / f"{file_id}_qc_report.xlsx"
        async with aiofiles.open(report_path, 'wb') as f:
            await f.write(xlsx_bytes)

        await websocket.send_json({
            "type": "complete",
            "download_url": f"/qc-report/{file_id}"
        })

    except WebSocketDisconnect:
        print(f"QC report WS disconnected for {file_id}")
    except Exception as e:
        try:
            await websocket.send_json({
                "type": "error", "message": f"Report generation error: {str(e)}"
            })
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@app.get("/qc-report/{file_id}")
async def qc_report(file_id: str):
    """
    Download a previously generated QC deviation report.
    """
    report_path = UPLOAD_DIR / f"{file_id}_qc_report.xlsx"

    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found. Generate it first.")

    return FileResponse(
        report_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=f"deviation_report_{file_id}.xlsx"
    )


@app.get("/qc-rerun/{file_id}")
async def qc_rerun(file_id: str):
    """
    Re-run QC deviation analysis using the previously uploaded CSV
    and the current (possibly updated) detections GeoJSON.
    """
    csv_path = UPLOAD_DIR / f"{file_id}_qc.csv"
    geojson_path = UPLOAD_DIR / f"{file_id}_detections.geojson"
    tif_path = UPLOAD_DIR / f"{file_id}.tif"

    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="No QC CSV found. Upload one first.")
    if not geojson_path.exists():
        raise HTTPException(status_code=404, detail="No detections found. Run inference first.")
    if not tif_path.exists():
        raise HTTPException(status_code=404, detail="Source TIF not found.")

    async with aiofiles.open(csv_path, 'rb') as f:
        csv_bytes = await f.read()

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: compute_deviations(csv_bytes, str(geojson_path), str(tif_path))
        )
        return JSONResponse(result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"CSV parsing error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"QC rerun error: {str(e)}")


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
