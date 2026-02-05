// Global state
let map = null;
let imageOverlay = null;
let detectionsLayer = null;
let currentFileId = null;
let ws = null;
let allDetections = null; // Store all detections GeoJSON
let selectedFeatures = new Set(); // Track selected feature indices
let deletedFeatures = new Set(); // Track deleted feature indices

// Progress timing state
let progressStartTime = null;
let reassuranceTimer = null;
let reassuranceIndex = 0;

const uploadReassuranceMessages = [
    "Large files take a bit longer — everything is running normally.",
    "Still working — generating the high-resolution overlay for your map.",
    "This is a good time to grab a coffee. We'll be here when you get back.",
    "No errors — just crunching through a lot of pixels.",
    "Almost there... large GeoTIFFs need extra time for reprojection.",
    "Still going strong. Hang tight."
];

const inferenceReassuranceMessages = [
    "The model is scanning your image tile by tile — this is normal.",
    "Large images have many tiles to process. The connection is healthy.",
    "Still running inference — grab a coffee, we'll keep at it.",
    "No errors — each tile takes a moment to run through the neural network.",
    "Progress will jump in bursts as tiles complete. Patience pays off.",
    "The WebSocket is alive and well. Processing continues in the background."
];

// Which message set to use
let activeReassuranceMessages = uploadReassuranceMessages;

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
    setStageIcon('upload');
    activeReassuranceMessages = uploadReassuranceMessages;
    updateProgress(0, 'Uploading file...');
    startReassuranceTimer();

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
                stopReassuranceTimer();
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
    stopReassuranceTimer();
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

// WebSocket indicator helpers
function showWsIndicator(connected) {
    const indicator = document.getElementById('ws-indicator');
    if (!indicator) return;
    indicator.style.display = 'inline-flex';
    if (connected) {
        indicator.classList.add('connected');
        indicator.querySelector('.ws-label').textContent = 'Connected';
    } else {
        indicator.classList.remove('connected');
        indicator.querySelector('.ws-label').textContent = 'Disconnected';
    }
}

function hideWsIndicator() {
    const indicator = document.getElementById('ws-indicator');
    if (indicator) {
        indicator.style.display = 'none';
        indicator.classList.remove('connected', 'ping');
    }
}

function flashWsPing() {
    const indicator = document.getElementById('ws-indicator');
    if (!indicator) return;
    indicator.classList.remove('ping');
    // Force reflow so the animation restarts
    void indicator.offsetWidth;
    indicator.classList.add('ping');
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

    // Start reassurance for long inference (use inference-specific messages)
    setStageIcon('processing');
    activeReassuranceMessages = inferenceReassuranceMessages;
    startReassuranceTimer();

    // Create WebSocket connection
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/inference/${fileId}`;
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log('WebSocket connected');
        showWsIndicator(true);
    };

    ws.onmessage = (event) => {
        const message = JSON.parse(event.data);
        handleWebSocketMessage(message);
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        showWsIndicator(false);
        alert('Connection error. Please try again.');
        resetInferenceUI();
    };

    ws.onclose = () => {
        console.log('WebSocket closed');
        hideWsIndicator();
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

        case 'ping':
            // Keep-alive from server — flash the connection dot
            flashWsPing();
            break;
    }
}

// Switch the active stage SVG icon
function setStageIcon(stage) {
    const icons = document.querySelectorAll('.stage-svg');
    icons.forEach(svg => svg.classList.remove('active'));

    const heading = document.getElementById('progress-heading');
    switch (stage) {
        case 'upload':
            document.getElementById('svg-upload').classList.add('active');
            heading.textContent = 'Uploading';
            break;
        case 'processing':
            document.getElementById('svg-processing').classList.add('active');
            heading.textContent = 'Processing';
            break;
        case 'overlay':
            document.getElementById('svg-overlay').classList.add('active');
            heading.textContent = 'Generating Overlay';
            break;
    }
}

// Start the reassurance message cycle
function startReassuranceTimer() {
    stopReassuranceTimer();
    progressStartTime = Date.now();
    reassuranceIndex = 0;
    const el = document.getElementById('progress-reassurance');
    if (el) el.textContent = '';

    // Show first message after 20 seconds, then cycle every 25s
    reassuranceTimer = setTimeout(function tick() {
        const el = document.getElementById('progress-reassurance');
        if (el && reassuranceIndex < activeReassuranceMessages.length) {
            el.textContent = activeReassuranceMessages[reassuranceIndex];
            reassuranceIndex++;
        } else if (el) {
            reassuranceIndex = 0;
            el.textContent = activeReassuranceMessages[reassuranceIndex];
            reassuranceIndex++;
        }
        reassuranceTimer = setTimeout(tick, 25000);
    }, 20000);
}

function stopReassuranceTimer() {
    if (reassuranceTimer) {
        clearTimeout(reassuranceTimer);
        reassuranceTimer = null;
    }
    const el = document.getElementById('progress-reassurance');
    if (el) el.textContent = '';
}

// Update progress bar
function updateProgress(percentage, text) {
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');

    progressFill.style.width = `${percentage}%`;
    progressText.textContent = text;

    // Auto-detect stage from the text and switch icon
    if (text.toLowerCase().includes('upload')) {
        setStageIcon('upload');
    } else if (text.toLowerCase().includes('overlay') || text.toLowerCase().includes('processing image')) {
        setStageIcon('overlay');
    } else if (text.toLowerCase().includes('tile') || text.toLowerCase().includes('inference') || text.toLowerCase().includes('merging')) {
        setStageIcon('processing');
    }
}

// Handle inference completion
function handleInferenceComplete(message) {
    // Stop reassurance messages
    stopReassuranceTimer();

    // 1. Hide the progress UI
    progressSection.style.display = 'none';

    // 2. Show the results section
    resultsSection.style.display = 'block';

    // 3. Mark progress as 100 % (nice for the user)
    updateProgress(100, 'Complete!');

    // 4. Render detections on the map
    if (message.detections && message.detections.features.length > 0) {
        addDetectionsToMap(message.detections);
        updateDetectionStats();
    } else {
        console.warn('No detections returned');
    }

    // Re-enable the inference button
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
            <strong>${remainingCount}</strong> objects detected
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
    stopReassuranceTimer();
    hideWsIndicator();
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
