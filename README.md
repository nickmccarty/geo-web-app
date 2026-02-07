# GeoTIFF Object Detection Web App

A FastAPI-powered web application for running object detection on GeoTIFF files using your trained PyTorch model, with built-in QC deviation analysis and Excel reporting.

<p align="center">
  <img
    src="https://github.com/nickmccarty/geo-web-app/raw/refs/heads/main/static/images/demo-video.gif"
    alt="Geo Web App demo"
    width="500"
  />
</p>

## Features

- **Drag & drop** GeoTIFF upload with progress tracking
- **Interactive map** with Leaflet.js and high-resolution overlay (up to 8192px)
- **Real-time inference** with tile-based progress via WebSocket
- **Model selection** - switch between `.pth` checkpoints from a dropdown
- **Device toggle** - run inference on CPU or GPU
- **Detection visualization** with interactive editing (select, delete, toggle)
- **Filtered GeoJSON export** with proper CRS preservation
- **Automatic geometry merging** for cleaner results
- **QC Deviation Analysis** - upload a CSV of check points and compare against detections
- **Excel Deviation Report** with embedded visualization and deviation snapshots
- **Auto-rerun QC** - switching models and re-running inference automatically refreshes QC results
- **Auto cleanup** of uploaded files after 1 hour

## Setup

### 1. Add Model Checkpoints

Place one or more `.pth` checkpoint files in the `checkpoints/` directory. The app will prefer `best_model.pth` on startup if present, otherwise it loads the first `.pth` file found.

### 2. Build & Run with Docker

```bash
docker build -t geo-web-app-cpu .
docker run --rm -p 8000:8000 geo-web-app-cpu
```

The Docker image includes all dependencies and runs inference on **CPU** — no CUDA setup required.

### 3. Open in Browser

Navigate to: **http://localhost:8000**

## Usage

1. **Upload**: Drag and drop a `.tif` or `.tiff` file or click to browse
2. **Preview**: View the image bounds and high-resolution overlay on the map
3. **Select Model/Device**: Choose a checkpoint from the dropdown and toggle CPU/GPU
4. **Run Inference**: Click "Run Inference" and watch the real-time progress
5. **View Results**: See detected objects overlaid on the map with live statistics
6. **Edit Detections** (optional):
   - Click polygons to select them (turns red)
   - Click again to deselect
   - Press **Esc** to clear all selections
   - Use "Delete Selected" button to remove false positives
7. **Toggle Layers**: Show/hide the image overlay or detections as needed
8. **Download GeoJSON**: Export filtered results (excludes deleted detections)

### QC Deviation Analysis

1. After running inference, click **Upload QC Points** and select a CSV file (a sample is included at `static/qc-points.csv`)
   - Supported formats: header-based (`point_id, y, x, z`) or positional columns
   - Coordinates should be in the TIF's native CRS (e.g., NAD83 State Plane, feet)
2. The map displays QC markers, centroid markers, connecting lines, and distance labels
3. A deviation table shows all matched points with click-to-zoom
4. Click **Download Report** to generate an Excel workbook with:
   - **Sheet 1**: Deviation table with status highlighting and an embedded visualization
   - **Sheet 2**: Detected centroids in WGS84
   - **Sheet 3**: Cropped snapshots for each deviation exceeding the 3cm threshold
5. If you switch models and re-run inference, QC results automatically refresh

## API Endpoints

### HTTP

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Main page |
| `POST` | `/upload` | Upload GeoTIFF file |
| `GET` | `/models` | List available checkpoints and current model state |
| `POST` | `/models/load` | Load a checkpoint on a specific device |
| `POST` | `/qc-analysis/{file_id}` | Run QC deviation analysis (accepts CSV upload) |
| `GET` | `/qc-rerun/{file_id}` | Re-run QC analysis with current detections |
| `GET` | `/qc-report/{file_id}` | Download generated Excel report |
| `GET` | `/download/{file_id}` | Download detection results as GeoJSON |
| `GET` | `/health` | Health check |

### WebSocket

| Path | Description |
|------|-------------|
| `WS /ws/process/{file_id}` | File processing (preview, overlay, bounds) with progress logs |
| `WS /ws/inference/{file_id}` | Real-time inference with tile progress |
| `WS /ws/qc-report/{file_id}` | Report generation with progress logs |

## Configuration

Edit configuration in `app.py`:

```python
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
```

Model parameters in `model_inference.py`:

```python
tile_size=1024,        # Tile size for inference
overlap=128,           # Overlap between tiles
score_threshold=0.5,   # Minimum confidence score
```

QC threshold in `utils.py`:

```python
THRESHOLD_CM = 3.0     # Deviation threshold in centimeters
FEET_TO_CM = 30.48     # Conversion factor (native CRS in feet)
```

## Troubleshooting

### Model Not Loading

Ensure at least one `.pth` checkpoint file exists in the `checkpoints/` directory before building the image.

### WebSocket Connection Failed

Check firewall settings and ensure port 8000 is accessible.

### Container Exits Immediately

Run with `-it` to see error output:

```bash
docker run --rm -it -p 8000:8000 geo-web-app-cpu
```

## Technologies Used

- **Backend**: FastAPI, PyTorch, Rasterio
- **Frontend**: Vanilla JavaScript, Leaflet.js
- **ML**: Faster R-CNN ResNet50-FPN
- **Geo**: GeoPandas, Shapely
- **Reporting**: openpyxl, matplotlib

## License

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
