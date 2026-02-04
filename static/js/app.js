// Global state
let map = null;
let imageOverlay = null;
let detectionsLayer = null;
let currentFileId = null;
let ws = null;
let allDetections = null; // Store all detections GeoJSON
let selectedFeatures = new Set(); // Track selected feature indices
let deletedFeatures = new Set(); // Track deleted feature indices

// DOM Elements
const uploadArea = document.getElementById('upload-area');
const fileInput = document.getElementById('file-input');
const uploadSection = document.getElementById('upload-section');
const previewSection = document.getElementById('preview-section');
const progressSection = document.getElementById('progress-section');
const resultsSection = document.getElementById('results-section');
const runInferenceBtn = document.getElementById('run-inference-btn');
const resetBtn = document.getElementById('reset-btn');
const downloadBtn = document.getElementById('download-btn');
const toggleDetectionsBtn = document.getElementById('toggle-detections-btn');
const toggleImageBtn = document.getElementById('toggle-image-btn');
const deleteSelectedBtn = document.getElementById('delete-selected-btn');

// Upload area interactions
uploadArea.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('drag-over');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileUpload(files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileUpload(e.target.files[0]);
    }
});

// Reset button
resetBtn.addEventListener('click', () => {
    location.reload();
});

// Toggle image overlay button
toggleImageBtn.addEventListener('click', () => {
    if (imageOverlay) {
        if (map.hasLayer(imageOverlay)) {
            map.removeLayer(imageOverlay);
            toggleImageBtn.textContent = '🖼️ Show Image';
        } else {
            map.addLayer(imageOverlay);
            toggleImageBtn.textContent = '🖼️ Hide Image';
        }
    }
});

// Run inference button
runInferenceBtn.addEventListener('click', () => {
    if (currentFileId) {
        runInference(currentFileId);
    }
});

// Download button
downloadBtn.addEventListener('click', () => {
    if (currentFileId && allDetections) {
        downloadFilteredGeoJSON();
    }
});

// Delete selected button
deleteSelectedBtn.addEventListener('click', () => {
    if (selectedFeatures.size > 0) {
        deleteSelectedFeatures();
    }
});

// Toggle detections button
toggleDetectionsBtn.addEventListener('click', () => {
    if (detectionsLayer) {
        if (map.hasLayer(detectionsLayer)) {
            map.removeLayer(detectionsLayer);
            toggleDetectionsBtn.textContent = '👁️ Show Detections';
        } else {
            map.addLayer(detectionsLayer);
            toggleDetectionsBtn.textContent = '🙈 Hide Detections';
        }
    }
});

// Handle file upload
function handleFileUpload(file) {
    if (!file.name.toLowerCase().endsWith('.tif') && !file.name.toLowerCase().endsWith('.tiff')) {
        alert('Please upload a .tif or .tiff file');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    // Show progress section
    uploadSection.style.display = 'none';
    progressSection.style.display = 'block';
    updateProgress(0, 'Uploading file...');

    const xhr = new XMLHttpRequest();
    let processingInterval = null;

    // Track upload progress (0-50%)
    xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable) {
            const uploadPercentage = Math.round((e.loaded / e.total) * 50); // Scale to 50%
            updateProgress(uploadPercentage, `Uploading... ${Math.round((e.loaded / e.total) * 100)}%`);
        }
    });

    // When upload completes, simulate processing progress (50-95%)
    xhr.upload.addEventListener('loadend', () => {
        let processingProgress = 50;
        updateProgress(processingProgress, 'Processing image and generating overlay...');

        // Simulate gradual progress while server processes
        processingInterval = setInterval(() => {
            if (processingProgress < 95) {
                processingProgress += 1;
                updateProgress(processingProgress, 'Processing image and generating overlay...');
            }
        }, 200); // Update every 200ms
    });

    // Handle completion
    xhr.addEventListener('load', () => {
        // Clear processing interval
        if (processingInterval) {
            clearInterval(processingInterval);
        }

        if (xhr.status === 200) {
            try {
                const data = JSON.parse(xhr.responseText);
                currentFileId = data.file_id;

                // Update to 100%
                updateProgress(100, 'Complete!');

                // Show preview after a brief delay
                setTimeout(() => {
                    progressSection.style.display = 'none';
                    displayPreview(data);
                }, 500);

            } catch (error) {
                console.error('Parse error:', error);
                alert('Error processing response');
                resetUploadUI();
            }
        } else {
            try {
                const error = JSON.parse(xhr.responseText);
                alert(`Upload failed: ${error.detail || 'Unknown error'}`);
            } catch {
                alert(`Upload failed with status: ${xhr.status}`);
            }
            resetUploadUI();
        }
    });

    // Handle errors
    xhr.addEventListener('error', () => {
        if (processingInterval) {
            clearInterval(processingInterval);
        }
        alert('Upload failed. Please try again.');
        resetUploadUI();
    });

    // Handle abort
    xhr.addEventListener('abort', () => {
        if (processingInterval) {
            clearInterval(processingInterval);
        }
        alert('Upload cancelled.');
        resetUploadUI();
    });

    // Send request
    xhr.open('POST', '/upload');
    xhr.send(formData);
}

// Reset upload UI
function resetUploadUI() {
    progressSection.style.display = 'none';
    uploadSection.style.display = 'block';
}

// Display image preview and metadata
function displayPreview(data) {
    // Hide upload, show preview
    uploadSection.style.display = 'none';
    previewSection.style.display = 'block';

    // Display metadata
    const metadata = document.getElementById('metadata');
    metadata.innerHTML = `
        <div><strong>Filename:</strong> ${data.filename}</div>
        <div><strong>Dimensions:</strong> ${data.metadata.width} × ${data.metadata.height} px</div>
        <div><strong>CRS:</strong> ${data.metadata.crs}</div>
    `;

    // Initialize map
    initializeMap(data);
}

// Initialize Leaflet map
function initializeMap(data) {
    if (map) {
        map.remove();
    }

    // Create map
    map = L.map('map', {
        zoomControl: true,
        attributionControl: false,
        maxZoom: 24
    });

    // Add base layer
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 24,
        maxNativeZoom: 19
    }).addTo(map);

    // Get bounds from GeoJSON
    const bounds = data.bounds.features[0].geometry.coordinates[0];
    const latLngs = bounds.map(coord => [coord[1], coord[0]]);

    // Fit map to bounds
    const leafletBounds = L.latLngBounds(latLngs);
    map.fitBounds(leafletBounds);

    // Add image overlay using high-resolution overlay
    if (imageOverlay) {
        map.removeLayer(imageOverlay);
    }

    // Use overlay_url if available, fallback to preview_url
    const overlayUrl = data.overlay_url || data.preview_url;
    imageOverlay = L.imageOverlay(overlayUrl, leafletBounds, {
        opacity: 0.7,
        interactive: false
    }).addTo(map);

    // Add bounds polygon for reference
    L.polygon(latLngs, {
        color: '#3388ff',
        weight: 2,
        fillOpacity: 0,
        dashArray: '5, 5'
    }).addTo(map);
}

// Run inference via WebSocket
function runInference(fileId) {
    // Disable button
    runInferenceBtn.disabled = true;
    document.getElementById('btn-text').style.display = 'none';
    document.getElementById('btn-spinner').style.display = 'inline-block';

    // Show progress section
    progressSection.style.display = 'block';
    resultsSection.style.display = 'none';

    // Create WebSocket connection
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/inference/${fileId}`;
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log('WebSocket connected');
    };

    ws.onmessage = (event) => {
        const message = JSON.parse(event.data);
        handleWebSocketMessage(message);
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        alert('Connection error. Please try again.');
        resetInferenceUI();
    };

    ws.onclose = () => {
        console.log('WebSocket closed');
    };
}

// Handle WebSocket messages
function handleWebSocketMessage(message) {
    switch (message.type) {
        case 'status':
            updateProgress(0, message.message);
            break;

        case 'progress':
            updateProgress(message.percentage, `Processing tile ${message.current}/${message.total}...`);
            break;

        case 'complete':
            handleInferenceComplete(message);
            break;

        case 'error':
            alert(`Error: ${message.message}`);
            resetInferenceUI();
            break;
    }
}

// Update progress bar
function updateProgress(percentage, text) {
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');

    progressFill.style.width = `${percentage}%`;
    progressText.textContent = text;
}

// Handle inference completion
function handleInferenceComplete(message) {
    // Hide progress
    progressSection.style.display = 'none';

    // Show results
    resultsSection.style.display = 'block';

    // Add detections to map
    if (message.detections && message.detections.features.length > 0) {
        addDetectionsToMap(message.detections);

        // Update stats (initial state)
        updateDetectionStats();
    } else {
        // No detections found
        const statsDiv = document.getElementById('results-stats');
        statsDiv.innerHTML = `
            <div class="stat-item">
                <strong>0</strong> objects detected
            </div>
        `;
    }

    // Reset button
    resetInferenceUI();
}

// Add detections to map
function addDetectionsToMap(geojson) {
    // Store detections globally
    allDetections = geojson;

    if (detectionsLayer) {
        map.removeLayer(detectionsLayer);
    }

    // Clear selection state
    selectedFeatures.clear();
    deletedFeatures.clear();

    detectionsLayer = L.geoJSON(geojson, {
        style: (feature) => getFeatureStyle(feature, false),
        onEachFeature: (feature, layer) => {
            const featureIndex = geojson.features.indexOf(feature);

            // Click to select/deselect
            layer.on('click', (e) => {
                L.DomEvent.stopPropagation(e);
                toggleFeatureSelection(featureIndex, layer);
            });

            // Popup with info
            if (feature.properties) {
                const props = feature.properties;
                const popupContent = `
                    <div style="padding: 8px;">
                        <strong>Detection</strong><br>
                        Class: ${props.class || 'object'}<br>
                        ${props.score ? `Score: ${(props.score * 100).toFixed(1)}%` : ''}
                    </div>
                `;
                layer.bindPopup(popupContent);
            }

            // Store feature index on layer
            layer.featureIndex = featureIndex;
        }
    }).addTo(map);

    // Fit map to detections
    map.fitBounds(detectionsLayer.getBounds());
}

// Get style for a feature based on selection state
function getFeatureStyle(feature, isSelected) {
    if (isSelected) {
        return {
            color: '#ff0000',
            weight: 4,
            fillOpacity: 0.4,
            fillColor: '#ff0000'
        };
    } else {
        return {
            color: '#00ff00',
            weight: 3,
            fillOpacity: 0.2,
            fillColor: '#00ff00'
        };
    }
}

// Toggle feature selection
function toggleFeatureSelection(featureIndex, layer) {
    if (selectedFeatures.has(featureIndex)) {
        // Deselect
        selectedFeatures.delete(featureIndex);
        layer.setStyle(getFeatureStyle(null, false));
    } else {
        // Select
        selectedFeatures.add(featureIndex);
        layer.setStyle(getFeatureStyle(null, true));
    }

    // Update delete button visibility
    updateDeleteButtonVisibility();
}

// Update delete button visibility based on selection
function updateDeleteButtonVisibility() {
    const deleteBtn = document.getElementById('delete-selected-btn');
    if (deleteBtn) {
        if (selectedFeatures.size > 0) {
            deleteBtn.style.display = 'inline-block';
            deleteBtn.textContent = `🗑️ Delete Selected (${selectedFeatures.size})`;
        } else {
            deleteBtn.style.display = 'none';
        }
    }
}

// Delete selected features
function deleteSelectedFeatures() {
    // Add selected features to deleted set
    selectedFeatures.forEach(index => {
        deletedFeatures.add(index);
    });

    // Clear selection
    selectedFeatures.clear();

    // Rebuild the layer without deleted features
    rebuildDetectionsLayer();

    // Update stats
    updateDetectionStats();
}

// Rebuild detections layer excluding deleted features
function rebuildDetectionsLayer() {
    if (!allDetections) return;

    // Remove existing layer
    if (detectionsLayer) {
        map.removeLayer(detectionsLayer);
    }

    // Filter out deleted features
    const filteredGeoJSON = {
        type: 'FeatureCollection',
        features: allDetections.features.filter((_, index) => !deletedFeatures.has(index))
    };

    // Recreate layer with filtered features
    detectionsLayer = L.geoJSON(filteredGeoJSON, {
        style: (feature) => getFeatureStyle(feature, false),
        onEachFeature: (feature, layer) => {
            // Get original index from allDetections
            const originalIndex = allDetections.features.indexOf(feature);

            // Click to select/deselect
            layer.on('click', (e) => {
                L.DomEvent.stopPropagation(e);
                toggleFeatureSelection(originalIndex, layer);
            });

            // Popup with info
            if (feature.properties) {
                const props = feature.properties;
                const popupContent = `
                    <div style="padding: 8px;">
                        <strong>Detection</strong><br>
                        Class: ${props.class || 'object'}<br>
                        ${props.score ? `Score: ${(props.score * 100).toFixed(1)}%` : ''}
                    </div>
                `;
                layer.bindPopup(popupContent);
            }

            // Store feature index on layer
            layer.featureIndex = originalIndex;
        }
    }).addTo(map);

    // Update delete button visibility
    updateDeleteButtonVisibility();
}

// Update detection stats
function updateDetectionStats() {
    const statsDiv = document.getElementById('results-stats');
    const remainingCount = allDetections.features.length - deletedFeatures.size;

    let statsHTML = `
        <div class="stat-item">
            <strong>${remainingCount}</strong> objects remaining
        </div>
    `;

    if (deletedFeatures.size > 0) {
        statsHTML += `
            <div class="stat-item" style="border-left-color: #dc3545;">
                <strong>${deletedFeatures.size}</strong> deleted
            </div>
        `;
    }

    statsDiv.innerHTML = statsHTML;
}

// Download filtered GeoJSON (excluding deleted features)
function downloadFilteredGeoJSON() {
    if (!allDetections) return;

    // Filter out deleted features
    const filteredGeoJSON = {
        type: 'FeatureCollection',
        features: allDetections.features.filter((_, index) => !deletedFeatures.has(index))
    };

    // Create blob and download
    const blob = new Blob([JSON.stringify(filteredGeoJSON, null, 2)], {
        type: 'application/geo+json'
    });

    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `detections_${currentFileId}.geojson`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Reset inference UI
function resetInferenceUI() {
    runInferenceBtn.disabled = false;
    document.getElementById('btn-text').style.display = 'inline';
    document.getElementById('btn-spinner').style.display = 'none';
}

// Loading overlay
function showLoading(show) {
    // Simple implementation - you can add a full overlay if needed
    if (show) {
        document.body.style.cursor = 'wait';
    } else {
        document.body.style.cursor = 'default';
    }
}
