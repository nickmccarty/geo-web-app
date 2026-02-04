# GeoTIFF Object Detection Web App

A FastAPI-powered web application for running object detection on GeoTIFF files using your trained PyTorch model.

## Features

- 📤 **Drag & drop** GeoTIFF upload with progress tracking
- 🗺️ **Interactive map** with Leaflet.js
- 🚀 **Real-time inference** with progress tracking via WebSocket
- 🎯 **Detection visualization** with overlays
- ✏️ **Interactive editing** - click to select/deselect detections
- 🗑️ **Delete unwanted detections** before export
- 📊 **Live statistics** showing remaining and deleted counts
- 🖼️ **High-resolution overlay** generation (up to 8192px)
- 👁️ **Toggle layers** - show/hide image and detections independently
- 📥 **Filtered GeoJSON export** with proper CRS preservation
- 🔄 **Automatic geometry merging** for cleaner results
- 🧹 **Auto cleanup** of uploaded files after 1 hour

## Setup

### 1. Install Dependencies

```bash
cd geo-web-app
pip install -r requirements.txt
```

### 2. Run the Application

```bash
python app.py
```

Or using uvicorn directly:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 3. Open in Browser

Navigate to: **http://localhost:8000**

## Usage

1. **Upload**: Drag and drop a `.tif` or `.tiff` file or click to browse (example image provided in `static/images/crop_38.tif`)
2. **Preview**: View the image bounds and high-resolution overlay on the map
3. **Run Inference**: Click "Run Inference" and watch the real-time progress
4. **View Results**: See detected objects overlaid on the map with live statistics
5. **Edit Detections** (optional):
   - Click polygons to select them (turns red)
   - Click again to deselect
   - Use "Delete Selected" button to remove false positives
   - View updated statistics showing remaining/deleted counts
6. **Toggle Layers**: Show/hide the image overlay or detections as needed
7. **Download**: Export filtered results as GeoJSON (excludes deleted detections)

## API Endpoints

### HTTP Endpoints

- `GET /` - Main page
- `POST /upload` - Upload GeoTIFF file
- `GET /download/{file_id}` - Download results as GeoJSON
- `GET /health` - Health check

### WebSocket Endpoint

- `WS /ws/inference/{file_id}` - Real-time inference with progress updates

## Configuration

Edit configuration in `app.py`:

```python
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "best_model.pth"

# High-resolution overlay settings (in upload endpoint)
format='JPEG',         # Output format
quality=90             # JPEG quality (1-100)
```

Model parameters in `model_inference.py`:

```python
tile_size=1024,        # Tile size for inference
overlap=128,           # Overlap between tiles
score_threshold=0.5,   # Minimum confidence score
```

Automatic cleanup settings in `app.py`:

```python
max_age_hours=1        # Delete uploaded files after 1 hour
```

## Interactive Editing Workflow

The app includes a powerful interactive editing feature to clean up detection results:

1. **Visual Feedback**:
   - Default detections appear in **green**
   - Selected detections turn **red**
   - Deleted items are tracked in live statistics

2. **Selection**:
   - Click any polygon to select/deselect it
   - Multiple selections are supported
   - Delete button shows count: "Delete Selected (3)"

3. **Deletion**:
   - Only affects the export, not the original detections
   - Statistics update to show remaining vs deleted counts
   - Can toggle detections layer visibility after deletion

4. **Export**:
   - Downloaded GeoJSON automatically excludes deleted features
   - Original full detection file is still saved on server at `/download/{file_id}`

## Troubleshooting

### Model Not Loading

Ensure your checkpoint file is at `checkpoints/best_model.pth` and contains the correct state dict.

### CUDA Out of Memory

Reduce `tile_size` or use CPU:

```python
device="cpu"  # In model_inference.py
```

### WebSocket Connection Failed

Check firewall settings and ensure port 8000 is accessible.

### Slow Inference

- Use GPU if available
- Reduce `tile_size` for faster processing (less accurate on edges)
- Increase `overlap` for better edge handling (slower)

## Production Deployment

For production, use:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 2
```

Consider:
- **Reverse proxy** (nginx)
- **HTTPS** with SSL certificates
- **Authentication** (add auth middleware)
- **Rate limiting** (slowapi)
- **Docker** containerization

## Technologies Used

- **Backend**: FastAPI, PyTorch, Rasterio
- **Frontend**: Vanilla JavaScript, Leaflet.js
- **ML**: Faster R-CNN ResNet50-FPN
- **Geo**: GeoPandas, Shapely

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
