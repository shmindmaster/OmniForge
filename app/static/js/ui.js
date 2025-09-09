// UI.js - Handle file upload, drag-drop, and 3D STL viewer

let selectedFile = null;
let scene, camera, renderer, controls;

// DOM elements
const dropArea = document.getElementById('dropArea');
const fileInput = document.getElementById('fileInput');
const processBtn = document.getElementById('processBtn');
const statusText = document.getElementById('statusText');
const spinner = document.getElementById('spinner');
const confidenceValue = document.getElementById('confidenceValue');
const resultsSection = document.getElementById('resultsSection');
const artifactsSection = document.getElementById('artifactsSection');
const overlayPreview = document.getElementById('overlayPreview');
const csvLink = document.getElementById('csvLink');
const stlLink = document.getElementById('stlLink');
const stlViewer = document.getElementById('stlViewer');

// Drag and drop handlers
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, preventDefaults, false);
    document.body.addEventListener(eventName, preventDefaults, false);
});

['dragenter', 'dragover'].forEach(eventName => {
    dropArea.addEventListener(eventName, highlight, false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, unhighlight, false);
});

dropArea.addEventListener('drop', handleDrop, false);

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function highlight() {
    dropArea.classList.add('dragover');
}

function unhighlight() {
    dropArea.classList.remove('dragover');
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles(files);
}

// File input handler
fileInput.addEventListener('change', (e) => {
    handleFiles(e.target.files);
});

// Process button handler
processBtn.addEventListener('click', processImage);

function handleFiles(files) {
    if (files.length > 0) {
        const file = files[0];
        if (file.type.startsWith('image/')) {
            selectedFile = file;
            processBtn.disabled = false;
            updateStatus(`Selected: ${file.name}`);
        } else {
            alert('Please select an image file.');
        }
    }
}

function updateStatus(message, showSpinner = false) {
    statusText.textContent = message;
    spinner.style.display = showSpinner ? 'block' : 'none';
}

async function processImage() {
    if (!selectedFile) return;
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    updateStatus('Processing...', true);
    processBtn.disabled = true;
    resultsSection.style.display = 'none';
    artifactsSection.style.display = 'none';
    
    try {
        const response = await fetch('/process', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Processing failed');
        }
        
        const result = await response.json();
        displayResults(result);
        
    } catch (error) {
        updateStatus(`Error: ${error.message}`);
        console.error('Processing error:', error);
    } finally {
        processBtn.disabled = false;
    }
}

function displayResults(result) {
    updateStatus('Processing complete');
    
    // Update confidence
    const confidence = result.scale_confidence;
    confidenceValue.textContent = confidence === 1.0 ? 'High' : 
                                 confidence === 0.0 ? 'Low' : 
                                 confidence.toFixed(2);
    
    // Display overlay preview
    overlayPreview.src = result.overlay_url;
    
    // Update metrics table
    const metrics = result.metrics || {};
    updateMetric('lengthMm', metrics.length_mm);
    updateMetric('widthProxMm', metrics.width_prox_mm);
    updateMetric('widthMidMm', metrics.width_mid_mm);
    updateMetric('widthDistMm', metrics.width_dist_mm);
    updateMetric('mmPerPx', metrics.mm_per_px);
    updateMetric('sharpness', metrics.sharpness);
    updateMetric('maskArea', metrics.mask_area_px);
    
    // Setup download links
    csvLink.href = result.csv_url;
    csvLink.download = 'nail_metrics.csv';
    stlLink.href = result.stl_url;
    stlLink.download = 'nail_model.stl';
    
    // Show results sections
    resultsSection.style.display = 'grid';
    artifactsSection.style.display = 'block';
    
    // Initialize 3D viewer
    if (result.stl_url) {
        initThree(stlViewer, result.stl_url);
    }
}

function updateMetric(elementId, value) {
    const element = document.getElementById(elementId);
    if (element) {
        if (typeof value === 'number') {
            element.textContent = value.toFixed(1);
        } else {
            element.textContent = value || '-';
        }
    }
}

// Three.js STL Viewer
function initThree(canvas, stlUrl) {
    // Clear existing scene
    while(canvas.firstChild) {
        canvas.removeChild(canvas.firstChild);
    }
    
    // Scene setup
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf0f0f0);
    
    camera = new THREE.PerspectiveCamera(75, canvas.width / canvas.height, 0.1, 1000);
    camera.position.set(5, 5, 5);
    
    renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
    renderer.setSize(canvas.width, canvas.height);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    
    // Lighting
    const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
    scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(10, 10, 5);
    directionalLight.castShadow = true;
    scene.add(directionalLight);
    
    // Controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    
    // Load STL
    const loader = new THREE.STLLoader();
    loader.load(stlUrl, function(geometry) {
        const material = new THREE.MeshPhongMaterial({ 
            color: 0x8B4B8A,
            shininess: 100
        });
        
        const mesh = new THREE.Mesh(geometry, material);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        
        // Center and scale the model
        geometry.computeBoundingBox();
        const box = geometry.boundingBox;
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        
        mesh.position.sub(center);
        const maxDim = Math.max(size.x, size.y, size.z);
        const scale = 2 / maxDim;
        mesh.scale.setScalar(scale);
        
        scene.add(mesh);
        
        // Adjust camera to fit the model
        camera.position.set(3, 3, 3);
        controls.target.set(0, 0, 0);
        controls.update();
        
    }, undefined, function(error) {
        console.error('Error loading STL:', error);
    });
    
    // Animation loop
    function animate() {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
    }
    animate();
}

// Copy URL functionality
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        // Could show a toast notification here
        console.log('URL copied to clipboard');
    }).catch(err => {
        console.error('Failed to copy URL:', err);
    });
}