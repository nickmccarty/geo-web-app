FROM python:3.10-slim

# System deps for GDAL/rasterio/opencv
RUN apt-get update && apt-get install -y \
    gdal-bin libgdal-dev libspatialindex-dev \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

WORKDIR /app
COPY . /app

# Install geo stack first (CPU)
RUN pip install --no-cache-dir \
    "numpy>=2.0" \
    "pandas>=2.2" \
    "rasterio>=1.4" \
    "geopandas>=1.0" \
    "shapely>=2.1" \
    "matplotlib>=3.10" \
    "pillow>=10" \
    "opencv-python-headless>=4.11"

# CPU-only torch stack
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# App deps (orthomasker + FastAPI, etc.)
RUN pip install --no-cache-dir \
    fastapi==0.109.0 \
    "uvicorn[standard]==0.27.0" \
    python-multipart==0.0.6 \
    websockets==12.0 \
    albumentations==1.3.1 \
    tqdm==4.66.1 \
    aiofiles==23.2.1 \
    openpyxl==3.1.2

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
