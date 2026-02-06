// Global state
let map = null;
let imageOverlay = null;
let detectionsLayer = null;
let currentFileId = null;
let ws = null;
let allDetections = null; // Store all detections GeoJSON
let selectedFeatures = new Set(); // Track selected feature indices
let deletedFeatures = new Set(); // Track deleted feature indices
let qcLayer = null; // Leaflet layer group for QC markers, lines, labels
let qcData = null; // QC analysis response from server

// SVG icon constants (Bootstrap Icons) for dynamic button updates
const ICON = {
    image: '<svg class="btn-icon" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M6.002 5.5a1.5 1.5 0 1 1-3 0 1.5 1.5 0 0 1 3 0"/><path d="M2.002 1a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V3a2 2 0 0 0-2-2zm12 1a1 1 0 0 1 1 1v6.5l-3.777-1.947a.5.5 0 0 0-.577.093l-3.71 3.71-2.66-1.772a.5.5 0 0 0-.63.062L1.002 12V3a1 1 0 0 1 1-1z"/></svg>',
    toggle: '<svg class="btn-icon" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M11 4a4 4 0 0 1 0 8H8a5 5 0 0 0 2-4 5 5 0 0 0-2-4zm-6 8a4 4 0 1 1 0-8 4 4 0 0 1 0 8M0 8a5 5 0 0 0 5 5h6a5 5 0 0 0 0-10H5a5 5 0 0 0-5 5"/></svg>',
    trash: '<svg class="btn-icon" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M5.5 5.5A.5.5 0 0 1 6 6v6a.5.5 0 0 1-1 0V6a.5.5 0 0 1 .5-.5m2.5 0a.5.5 0 0 1 .5.5v6a.5.5 0 0 1-1 0V6a.5.5 0 0 1 .5-.5m3 .5a.5.5 0 0 0-1 0v6a.5.5 0 0 0 1 0z"/><path d="M14.5 3a1 1 0 0 1-1 1H13v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V4h-.5a1 1 0 0 1-1-1V2a1 1 0 0 1 1-1H6a1 1 0 0 1 1-1h2a1 1 0 0 1 1 1h3.5a1 1 0 0 1 1 1zM4.118 4 4 4.059V13a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1V4.059L11.882 4zM2.5 3h11V2h-11z"/></svg>',
    upload: '<svg class="btn-icon" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M.5 9.9a.5.5 0 0 1 .5.5v2.5a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-2.5a.5.5 0 0 1 1 0v2.5a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2v-2.5a.5.5 0 0 1 .5-.5"/><path d="M7.646 1.146a.5.5 0 0 1 .708 0l3 3a.5.5 0 0 1-.708.708L8.5 2.707V11.5a.5.5 0 0 1-1 0V2.707L5.354 4.854a.5.5 0 1 1-.708-.708z"/></svg>',
    slashCircle: '<svg class="btn-icon" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M8 15A7 7 0 1 1 8 1a7 7 0 0 1 0 14m0 1A8 8 0 1 0 8 0a8 8 0 0 0 0 16"/><path d="M11.354 4.646a.5.5 0 0 0-.708 0l-6 6a.5.5 0 0 0 .708.708l6-6a.5.5 0 0 0 0-.708"/></svg>',
};

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
const uploadQcBtn = document.getElementById('upload-qc-btn');
const qcFileInput = document.getElementById('qc-file-input');
const qcSection = document.getElementById('qc-section');
const toggleQcBtn = document.getElementById('toggle-qc-btn');
const downloadQcReportBtn = document.getElementById('download-qc-report-btn');

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
            toggleImageBtn.innerHTML = `${ICON.image} <span>Show Image</span>`;
        } else {
            map.addLayer(imageOverlay);
            toggleImageBtn.innerHTML = `${ICON.image} <span>Hide Image</span>`;
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
            toggleDetectionsBtn.innerHTML = `${ICON.toggle} <span>Show Detections</span>`;
        } else {
            map.addLayer(detectionsLayer);
            toggleDetectionsBtn.innerHTML = `${ICON.toggle} <span>Hide Detections</span>`;
        }
    }
});

// QC upload button
uploadQcBtn.addEventListener('click', () => qcFileInput.click());

qcFileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        uploadQcPoints(e.target.files[0]);
    }
});

// Toggle QC layer button
toggleQcBtn.addEventListener('click', () => {
    if (qcLayer) {
        if (map.hasLayer(qcLayer)) {
            map.removeLayer(qcLayer);
            toggleQcBtn.innerHTML = `${ICON.slashCircle} <span>Show QC Layer</span>`;
        } else {
            map.addLayer(qcLayer);
            toggleQcBtn.innerHTML = `${ICON.slashCircle} <span>Hide QC Layer</span>`;
        }
    }
});

// Download QC report button
downloadQcReportBtn.addEventListener('click', () => {
    if (currentFileId) {
        generateQcReport(currentFileId);
    }
});

// --- Log console helpers ---
function appendLog(message, type = 'info') {
    const entries = document.getElementById('log-entries');
    if (!entries) return;

    const entry = document.createElement('div');
    entry.className = `log-entry log-${type}`;

    const now = new Date();
    const ts = now.toLocaleTimeString('en-US', { hour12: false });

    entry.innerHTML = `<span class="log-time">${ts}</span><span class="log-msg">${message}</span>`;
    entries.appendChild(entry);

    // Auto-scroll to bottom
    entries.scrollTop = entries.scrollHeight;
}

function clearLog() {
    const entries = document.getElementById('log-entries');
    if (entries) entries.innerHTML = '';
}

// Track the original filename across the two-phase upload
let uploadedFilename = null;

// Handle file upload (two-phase: XHR save → WS process)
function handleFileUpload(file) {
    if (!file.name.toLowerCase().endsWith('.tif') && !file.name.toLowerCase().endsWith('.tiff')) {
        alert('Please upload a .tif or .tiff file');
        return;
    }

    uploadedFilename = file.name;
    const formData = new FormData();
    formData.append('file', file);

    // Show progress section
    uploadSection.style.display = 'none';
    progressSection.style.display = 'block';
    setStageIcon('upload');
    activeReassuranceMessages = uploadReassuranceMessages;
    clearLog();
    updateProgress(0, 'Uploading file...');
    appendLog('Starting upload...', 'status');
    startReassuranceTimer();

    const xhr = new XMLHttpRequest();

    // Track upload progress (0-50%)
    xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable) {
            const pct = Math.round((e.loaded / e.total) * 100);
            const barPct = Math.round((e.loaded / e.total) * 50);
            updateProgress(barPct, `Uploading... ${pct}%`);
        }
    });

    xhr.upload.addEventListener('loadend', () => {
        updateProgress(50, 'File saved on server. Connecting...');
        appendLog('Upload complete. File saved on server.', 'status');
    });

    // On XHR success → open WebSocket for processing
    xhr.addEventListener('load', () => {
        if (xhr.status === 200) {
            try {
                const data = JSON.parse(xhr.responseText);
                currentFileId = data.file_id;
                appendLog(`File ID: ${data.file_id}`, 'info');
                startProcessingWebSocket(data.file_id);
            } catch (error) {
                console.error('Parse error:', error);
                appendLog('Error parsing server response.', 'error');
                alert('Error processing response');
                resetUploadUI();
            }
        } else {
            try {
                const error = JSON.parse(xhr.responseText);
                appendLog(`Upload failed: ${error.detail || 'Unknown error'}`, 'error');
                alert(`Upload failed: ${error.detail || 'Unknown error'}`);
            } catch {
                appendLog(`Upload failed with status ${xhr.status}`, 'error');
                alert(`Upload failed with status: ${xhr.status}`);
            }
            resetUploadUI();
        }
    });

    xhr.addEventListener('error', () => {
        appendLog('Network error during upload.', 'error');
        alert('Upload failed. Please try again.');
        resetUploadUI();
    });

    xhr.addEventListener('abort', () => {
        appendLog('Upload cancelled.', 'error');
        alert('Upload cancelled.');
        resetUploadUI();
    });

    xhr.open('POST', '/upload');
    xhr.send(formData);
}

// Phase 2: process the uploaded file via WebSocket (preview, overlay, bounds)
function startProcessingWebSocket(fileId) {
    setStageIcon('overlay');
    updateProgress(52, 'Processing file...');

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/process/${fileId}`;
    const processWs = new WebSocket(wsUrl);

    // Track log messages for progress bar (server sends ~7 log messages)
    let logCount = 0;
    const totalExpectedLogs = 7;

    processWs.onopen = () => {
        showWsIndicator(true);
        appendLog('WebSocket connected. Processing starting...', 'status');
    };

    processWs.onmessage = (event) => {
        const msg = JSON.parse(event.data);

        switch (msg.type) {
            case 'log':
                logCount++;
                appendLog(msg.message, 'info');
                // Map log count to progress 52-95%
                const pct = Math.min(52 + Math.round((logCount / totalExpectedLogs) * 43), 95);
                updateProgress(pct, msg.message);
                break;

            case 'ping':
                flashWsPing();
                break;

            case 'complete':
                appendLog('Processing complete.', 'status');
                updateProgress(100, 'Complete!');
                stopReassuranceTimer();
                // Build the data object displayPreview expects
                const previewData = {
                    file_id: msg.file_id,
                    filename: uploadedFilename,
                    preview_url: msg.preview_url,
                    overlay_url: msg.overlay_url,
                    metadata: msg.metadata,
                    bounds: msg.bounds
                };
                setTimeout(() => {
                    hideWsIndicator();
                    progressSection.style.display = 'none';
                    displayPreview(previewData);
                }, 800);
                break;

            case 'error':
                appendLog(`Error: ${msg.message}`, 'error');
                alert(`Processing error: ${msg.message}`);
                resetUploadUI();
                break;
        }
    };

    processWs.onerror = () => {
        showWsIndicator(false);
        appendLog('WebSocket connection error.', 'error');
        alert('Processing connection error. Please try again.');
        resetUploadUI();
    };

    processWs.onclose = () => {
        hideWsIndicator();
    };
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

    // Populate model controls
    fetchModelState();
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
        maxZoom: 28
    });

    // Add base layer
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 28,
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
    clearLog();
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

// Handle WebSocket messages (inference)
function handleWebSocketMessage(message) {
    switch (message.type) {
        case 'status': {
            appendLog(message.message, 'status');
            const msg = message.message.toLowerCase();
            if (msg.includes('merging') || msg.includes('finalizing')) {
                // Switch to merging stage with indeterminate bar
                setStageIcon('merging');
                document.getElementById('progress-text').textContent = message.message;
            } else {
                updateProgress(0, message.message);
            }
            break;
        }

        case 'log':
            appendLog(message.message, 'info');
            break;

        case 'progress':
            updateProgress(message.percentage, `Processing tile ${message.current}/${message.total}...`);
            // Log every 10th tile to avoid flooding the console
            if (message.current === 1 || message.current === message.total || message.current % 10 === 0) {
                appendLog(`Tile ${message.current}/${message.total} (${message.percentage}%)`, 'progress');
            }
            break;

        case 'complete':
            appendLog('Inference complete.', 'status');
            handleInferenceComplete(message);
            break;

        case 'error':
            appendLog(`Error: ${message.message}`, 'error');
            alert(`Error: ${message.message}`);
            resetInferenceUI();
            break;

        case 'ping':
            flashWsPing();
            break;
    }
}

// Switch the active stage SVG icon
function setStageIcon(stage) {
    const icons = document.querySelectorAll('.stage-svg');
    icons.forEach(svg => svg.classList.remove('active'));

    const heading = document.getElementById('progress-heading');
    const bar = document.querySelector('.progress-bar');

    // Remove indeterminate unless we're entering the merging stage
    if (stage !== 'merging' && bar) bar.classList.remove('indeterminate');

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
        case 'merging':
            document.getElementById('svg-merging').classList.add('active');
            heading.textContent = 'Merging Detections';
            if (bar) bar.classList.add('indeterminate');
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
    } else if (text.toLowerCase().includes('tile') || text.toLowerCase().includes('inference')) {
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
        // Show QC upload button now that detections are available
        uploadQcBtn.style.display = 'inline-flex';
    } else {
        console.warn('No detections returned');
    }

    // Re-enable the inference button
    resetInferenceUI();

    // Auto-rerun QC analysis if a CSV was previously uploaded
    if (qcData !== null && currentFileId) {
        fetch(`/qc-rerun/${currentFileId}`)
            .then(r => {
                if (!r.ok) return r.json().then(err => { throw new Error(err.detail); });
                return r.json();
            })
            .then(data => {
                qcData = data;
                displayQcResults(data);
            })
            .catch(err => {
                console.warn('QC rerun failed:', err.message);
            });
    }
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

// Clear all selections (Esc key)
function clearSelection() {
    if (selectedFeatures.size === 0) return;
    selectedFeatures.clear();
    // Reset all layer styles back to default
    if (detectionsLayer) {
        detectionsLayer.eachLayer(layer => {
            layer.setStyle(getFeatureStyle(null, false));
        });
    }
    updateDeleteButtonVisibility();
}

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        clearSelection();
    }
});

// Update delete button visibility based on selection
function updateDeleteButtonVisibility() {
    const deleteBtn = document.getElementById('delete-selected-btn');
    if (deleteBtn) {
        if (selectedFeatures.size > 0) {
            deleteBtn.style.display = 'inline-flex';
            deleteBtn.innerHTML = `${ICON.trash} <span>Delete Selected (${selectedFeatures.size})</span>`;
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

// --- Model Selection & Device Toggle ---

let modelLoading = false;

function fetchModelState() {
    fetch('/models')
        .then(r => r.json())
        .then(data => {
            populateModelDropdown(data.checkpoints, data.current_checkpoint);
            setActiveDevice(data.current_device, data.cuda_available);
            setModelStatus('Loaded', 'loaded');
        })
        .catch(() => setModelStatus('Error fetching models', 'error'));
}

function populateModelDropdown(checkpoints, currentCheckpoint) {
    const select = document.getElementById('model-select');
    select.innerHTML = '';
    checkpoints.forEach(name => {
        const opt = document.createElement('option');
        opt.value = name;
        opt.textContent = name.replace('.pth', '');
        if (name === currentCheckpoint) opt.selected = true;
        select.appendChild(opt);
    });
}

function setActiveDevice(device, cudaAvailable) {
    const buttons = document.querySelectorAll('.device-btn');
    buttons.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.device === device);
        if (btn.dataset.device === 'cuda') {
            btn.disabled = !cudaAvailable;
        }
    });
}

function setModelStatus(text, cls) {
    const status = document.getElementById('model-status');
    status.textContent = text;
    status.className = 'model-status' + (cls ? ' ' + cls : '');
}

function loadModel(checkpoint, device) {
    if (modelLoading) return;
    modelLoading = true;
    setModelStatus('Loading...', 'loading');
    runInferenceBtn.disabled = true;

    fetch('/models/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ checkpoint, device })
    })
    .then(r => {
        if (!r.ok) return r.json().then(err => { throw new Error(err.detail); });
        return r.json();
    })
    .then(data => {
        setActiveDevice(data.device, true);
        setModelStatus('Loaded', 'loaded');
    })
    .catch(err => {
        setModelStatus('Error', 'error');
        alert(`Failed to load model: ${err.message}`);
        // Re-fetch to restore correct UI state
        fetchModelState();
    })
    .finally(() => {
        modelLoading = false;
        runInferenceBtn.disabled = false;
    });
}

// Model dropdown change
document.getElementById('model-select').addEventListener('change', (e) => {
    const activeBtn = document.querySelector('.device-btn.active');
    loadModel(e.target.value, activeBtn ? activeBtn.dataset.device : 'cpu');
});

// Device toggle buttons
document.querySelectorAll('.device-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        if (btn.disabled || btn.classList.contains('active')) return;
        const select = document.getElementById('model-select');
        loadModel(select.value, btn.dataset.device);
    });
});

// --- QC Deviation Analysis ---

function uploadQcPoints(file) {
    if (!file.name.toLowerCase().endsWith('.csv')) {
        alert('Please upload a .csv file');
        return;
    }

    uploadQcBtn.disabled = true;
    uploadQcBtn.innerHTML = `${ICON.upload} <span>Analyzing...</span>`;

    const formData = new FormData();
    formData.append('file', file);

    fetch(`/qc-analysis/${currentFileId}`, {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => { throw new Error(err.detail || 'QC analysis failed'); });
        }
        return response.json();
    })
    .then(data => {
        qcData = data;
        displayQcResults(data);
    })
    .catch(error => {
        alert(`QC Analysis Error: ${error.message}`);
    })
    .finally(() => {
        uploadQcBtn.disabled = false;
        uploadQcBtn.innerHTML = `${ICON.upload} <span>Upload QC Points</span>`;
        qcFileInput.value = '';
    });
}

function displayQcResults(data) {
    qcSection.style.display = 'block';
    renderQcStats(data.summary);
    addQcLayerToMap(data.qc_points);
    renderDeviationTable(data.qc_points);
    toggleQcBtn.innerHTML = `${ICON.slashCircle} <span>Hide QC Layer</span>`;
}

function renderQcStats(summary) {
    const statsDiv = document.getElementById('qc-stats');
    const exceedColor = summary.count_exceeding_3cm > 0 ? '#dc3545' : '#28a745';

    statsDiv.innerHTML = `
        <div class="stat-item">
            <strong>${summary.matched_points}</strong> matched
        </div>
        <div class="stat-item">
            <strong>${summary.avg_deviation_cm.toFixed(2)}</strong> cm avg
        </div>
        <div class="stat-item">
            <strong>${summary.max_deviation_cm.toFixed(2)}</strong> cm max
        </div>
        <div class="stat-item" style="border-left-color: ${exceedColor};">
            <strong>${summary.count_exceeding_3cm}</strong> &gt; 3cm
        </div>
    `;
}

function addQcLayerToMap(qcPoints) {
    // Remove previous QC layer if re-uploading
    if (qcLayer) {
        map.removeLayer(qcLayer);
    }

    qcLayer = L.layerGroup();

    qcPoints.forEach(pt => {
        if (pt.lat == null || pt.lng == null) return;

        const isExceed = pt.exceeds_threshold;
        const lineColor = pt.matched ? (isExceed ? '#dc3545' : '#28a745') : '#888';

        // QC point marker (always white)
        const qcMarker = L.circleMarker([pt.lat, pt.lng], {
            radius: 6,
            fillColor: '#ffffff',
            color: '#000000',
            weight: 1.5,
            fillOpacity: 0.9
        });

        let tooltip = `QC #${pt.point_id}`;
        if (pt.matched && pt.deviation_cm != null) {
            tooltip += ` | ${pt.deviation_cm.toFixed(2)} cm`;
        } else if (!pt.matched) {
            tooltip += ' | No match';
        }
        qcMarker.bindTooltip(tooltip, { direction: 'top', offset: [0, -8] });
        qcLayer.addLayer(qcMarker);

        // If matched: centroid marker + connecting line + distance label
        if (pt.matched && pt.centroid_lat != null && pt.centroid_lng != null) {
            // Centroid marker (green or red based on threshold)
            const centroidMarker = L.circleMarker([pt.centroid_lat, pt.centroid_lng], {
                radius: 4,
                fillColor: lineColor,
                color: isExceed ? '#8b0000' : '#006400',
                weight: 1,
                fillOpacity: 0.85
            });
            qcLayer.addLayer(centroidMarker);

            // Dotted connecting line
            const line = L.polyline(
                [[pt.lat, pt.lng], [pt.centroid_lat, pt.centroid_lng]],
                {
                    color: lineColor,
                    weight: 2,
                    dashArray: '6, 4',
                    opacity: 0.8
                }
            );
            qcLayer.addLayer(line);

            // Distance label at midpoint
            const midLat = (pt.lat + pt.centroid_lat) / 2;
            const midLng = (pt.lng + pt.centroid_lng) / 2;
            const labelColor = isExceed ? '#ff4444' : '#44ff44';
            const labelHtml = `<span class="qc-distance-label" style="color:${labelColor}">${pt.deviation_cm.toFixed(2)} cm</span>`;

            const labelMarker = L.marker([midLat, midLng], {
                icon: L.divIcon({
                    className: 'qc-label-icon',
                    html: labelHtml,
                    iconSize: [100, 24],
                    iconAnchor: [110, 12]
                }),
                interactive: false
            });
            qcLayer.addLayer(labelMarker);
        }
    });

    qcLayer.addTo(map);
}

function renderDeviationTable(qcPoints) {
    const container = document.getElementById('qc-table-container');
    const matched = qcPoints.filter(p => p.matched);

    if (matched.length === 0) {
        container.innerHTML = '<p class="text-muted" style="padding: 20px;">No QC points matched any detection polygons.</p>';
        return;
    }

    matched.sort((a, b) => a.point_id - b.point_id);

    let html = '<table class="qc-table"><thead><tr>';
    html += '<th>Point ID</th><th>Deviation (cm)</th><th>Status</th>';
    html += '</tr></thead><tbody>';

    matched.forEach(pt => {
        const rowClass = pt.exceeds_threshold ? 'qc-row-exceed' : '';
        const status = pt.exceeds_threshold ? 'EXCEED' : 'OK';
        html += `<tr class="${rowClass}" data-lat="${pt.lat}" data-lng="${pt.lng}" style="cursor:pointer;">`;
        html += `<td>${pt.point_id}</td>`;
        html += `<td>${pt.deviation_cm.toFixed(2)}</td>`;
        html += `<td>${status}</td>`;
        html += '</tr>';
    });

    html += '</tbody></table>';
    container.innerHTML = html;

    // Click row to zoom to that QC point on the map
    container.querySelectorAll('tbody tr').forEach(row => {
        row.addEventListener('click', () => {
            const lat = parseFloat(row.dataset.lat);
            const lng = parseFloat(row.dataset.lng);
            if (!isNaN(lat) && !isNaN(lng)) {
                map.setView([lat, lng], 28, { animate: true });
            }
        });
    });
}

// --- QC Report Generation (WebSocket) ---

function generateQcReport(fileId) {
    const btnIcon = document.getElementById('report-btn-icon');
    const btnSpinner = document.getElementById('report-btn-spinner');
    const btnText = document.getElementById('report-btn-text');
    const logConsole = document.getElementById('qc-report-log');
    const logEntries = document.getElementById('qc-report-log-entries');

    // Show spinner, hide icon, disable button
    btnIcon.style.display = 'none';
    btnSpinner.style.display = 'inline-block';
    btnText.textContent = 'Generating...';
    downloadQcReportBtn.disabled = true;

    // Show and clear log console
    logConsole.style.display = 'block';
    logEntries.innerHTML = '';

    function appendReportLog(message) {
        const entry = document.createElement('div');
        entry.className = 'log-entry log-info';
        const now = new Date();
        const ts = now.toLocaleTimeString('en-US', { hour12: false });
        entry.innerHTML = `<span class="log-time">${ts}</span><span class="log-msg">${message}</span>`;
        logEntries.appendChild(entry);
        logEntries.scrollTop = logEntries.scrollHeight;
    }

    function resetReportBtn() {
        btnIcon.style.display = 'inline-flex';
        btnSpinner.style.display = 'none';
        btnText.textContent = 'Download Report';
        downloadQcReportBtn.disabled = false;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/qc-report/${fileId}`;
    const reportWs = new WebSocket(wsUrl);

    reportWs.onopen = () => {
        appendReportLog('Connected. Starting report generation...');
    };

    reportWs.onmessage = (event) => {
        const msg = JSON.parse(event.data);

        switch (msg.type) {
            case 'log':
                appendReportLog(msg.message);
                break;

            case 'complete':
                appendReportLog('Report ready. Starting download...');
                resetReportBtn();
                // Trigger download
                window.location.href = msg.download_url;
                // Fade out log after a delay
                setTimeout(() => {
                    logConsole.style.display = 'none';
                }, 3000);
                break;

            case 'error':
                appendReportLog(`Error: ${msg.message}`);
                resetReportBtn();
                alert(`Report Error: ${msg.message}`);
                break;
        }
    };

    reportWs.onerror = () => {
        appendReportLog('WebSocket connection error.');
        resetReportBtn();
        alert('Report generation connection error.');
    };

    reportWs.onclose = () => {};
}
