/* ============================================================
   CloudClassify13 - JavaScript
   ============================================================ */

// Cloud type descriptions
const CLOUD_DESCRIPTIONS = {
    'Ci': {
        name: 'Cirrus',
        description: 'Nubes altas, finas, cristalinas. Tienen forma filamentosa y se encuentran a más de 6000m de altura. Compuestas por cristales de hielo.'
    },
    'Cc': {
        name: 'Cirrocumulus',
        description: 'Nubes altas en forma de copos o grupos. Indican cambios de clima en las próximas 24 horas.'
    },
    'Cs': {
        name: 'Cirrostratus',
        description: 'Capas delgadas y translúcidas a gran altitud. Producen halos alrededor del sol o la luna.'
    },
    'Ac': {
        name: 'Altocumulus',
        description: 'Nubes medianas en racimos. Indican clima inestable y posibles tormentas.'
    },
    'As': {
        name: 'Altostratus',
        description: 'Capas grises o azuladas uniformes. El sol se ve como a través de vidrio esmerilado.'
    },
    'Cu': {
        name: 'Cumulus',
        description: 'Nubes densas con base plana y cúspides redondeadas. Indican buen tiempo.'
    },
    'Cb': {
        name: 'Cumulonimbus',
        description: 'Nubes de tormenta con desarrollo vertical. Producen lluvia, granizo y tormentas.'
    },
    'Ns': {
        name: 'Nimbostratus',
        description: 'Nubes bajas, oscuras y uniformes. Producen lluvia o nieve continua.'
    },
    'Sc': {
        name: 'Stratocumulus',
        description: 'Nubes bajas en capas o grupos. Ocupan gran parte del cielo.'
    },
    'St': {
        name: 'Stratus',
        description: 'Capas bajas, grises y uniformes. Pueden producir llovizna.'
    },
    'Ct': {
        name: 'Contrails',
        description: 'Estelas de condensación de aviones. Formadas a gran altitud.'
    }
};

// DOM Elements
const uploadArea = document.getElementById('upload-area');
const fileInput = document.getElementById('file-input');
const uploadSection = document.getElementById('upload-section');
const resultsSection = document.getElementById('results-section');
const resultContent = document.getElementById('result-content');

// Event Listeners
uploadArea.addEventListener('click', () => fileInput.click());
uploadArea.addEventListener('dragover', handleDragOver);
uploadArea.addEventListener('dragleave', handleDragLeave);
uploadArea.addEventListener('drop', handleDrop);
fileInput.addEventListener('change', handleFileSelect);

// Drag and Drop Handlers
function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.style.borderColor = '#1e40af';
    uploadArea.style.background = '#eff6ff';
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.style.borderColor = '#2563eb';
    uploadArea.style.background = '#f3f4f6';
}

function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.style.borderColor = '#2563eb';
    uploadArea.style.background = '#f3f4f6';

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        fileInput.files = files;
        processFile(files[0]);
    }
}

function handleFileSelect(e) {
    if (e.target.files.length > 0) {
        processFile(e.target.files[0]);
    }
}

// File Processing
function processFile(file) {
    // Validate file
    const validTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/gif'];
    const maxSize = 10 * 1024 * 1024; // 10MB

    if (!validTypes.includes(file.type)) {
        showError('Formato de archivo no válido. Use JPG, PNG, WebP o GIF.');
        return;
    }

    if (file.size > maxSize) {
        showError('El archivo es muy grande. Máximo 10MB.');
        return;
    }

    // Show loading
    showLoading();

    // Send to server
    uploadImage(file);
}

function uploadImage(file) {
    const formData = new FormData();
    formData.append('file', file);

    fetch('/api/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.success) {
            displayResults(data);
        } else {
            showError(data.error || 'Error en la predicción');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showError(`Error al conectar con el servidor: ${error.message}`);
    });
}

// Display Results
function displayResults(data) {
    const prediction = data.prediction;
    const confidence = data.confidence;
    const top_k = data.top_k || [];

    // Main prediction
    const cloudInfo = CLOUD_DESCRIPTIONS[prediction] || { name: prediction, description: '' };
    
    let html = `
        <div class="result-main">
            <div class="result-class">
                <div class="label">Predicción Principal</div>
                <h3>${prediction}</h3>
                <div class="class-code">${cloudInfo.name}</div>
            </div>

            <div class="confidence-bar">
                <div class="confidence-fill" style="width: ${confidence * 100}%"></div>
            </div>
            <div class="confidence-text">Confianza: ${(confidence * 100).toFixed(1)}%</div>
        </div>

        <div class="result-description">
            <h4><i class="fas fa-cloud"></i> Descripción</h4>
            <p>${cloudInfo.description}</p>
        </div>

        <div class="result-detailed">
            <h4><i class="fas fa-list"></i> Top 3 Predicciones</h4>
            <div class="top-predictions">
    `;

    top_k.forEach((item, index) => {
        const itemCode = item.class;
        const itemConf = item.confidence;
        const isActive = index === 0 ? 'active' : '';
        
        html += `
            <div class="prediction-item ${isActive}">
                <div class="prediction-class">${itemCode}</div>
                <div class="prediction-bar">
                    <div class="prediction-bar-fill" style="width: ${itemConf * 100}%"></div>
                </div>
                <div class="prediction-percent">${(itemConf * 100).toFixed(1)}%</div>
            </div>
        `;
    });

    html += `
            </div>
        </div>

        <button class="btn btn-secondary" onclick="resetUpload()">
            <i class="fas fa-redo"></i> Nueva Clasificación
        </button>
    `;

    resultContent.innerHTML = html;
    resultsSection.style.display = 'block';
    uploadArea.style.display = 'none';
    
    // Animate confidence bar
    setTimeout(() => {
        const fill = document.querySelector('.confidence-fill');
        if (fill) {
            fill.style.width = `${confidence * 100}%`;
        }
    }, 100);
}

// Show Error
function showError(message) {
    resultsSection.style.display = 'block';
    resultContent.innerHTML = `
        <div class="alert alert-error">
            <i class="fas fa-exclamation-circle"></i>
            <span>${message}</span>
        </div>
        <button class="btn btn-secondary" onclick="resetUpload()">
            <i class="fas fa-redo"></i> Volver a Intentar
        </button>
    `;
    uploadArea.style.display = 'none';
}

// Show Loading
function showLoading() {
    resultsSection.style.display = 'block';
    resultContent.innerHTML = `
        <div class="loading">
            <div class="spinner"></div>
            <p>Analizando imagen...</p>
        </div>
    `;
    uploadArea.style.display = 'none';
}

// Reset Upload
function resetUpload() {
    uploadArea.style.display = 'block';
    resultsSection.style.display = 'none';
    resultContent.innerHTML = '';
    fileInput.value = '';
    
    // Reset upload area style
    uploadArea.style.borderColor = '#2563eb';
    uploadArea.style.background = '#f3f4f6';
}

// Load Cloud Classes Info
async function loadCloudTypes() {
    try {
        const response = await fetch('/api/classes');
        if (!response.ok) throw new Error('Failed to load cloud types');
        
        const data = await response.json();
        displayCloudTypes(data.classes);
    } catch (error) {
        console.error('Error loading cloud types:', error);
    }
}

function displayCloudTypes(classes) {
    const cloudTypesGrid = document.querySelector('.cloud-types-grid');
    if (!cloudTypesGrid) return;

    let html = '';
    classes.forEach(cloudClass => {
        const info = CLOUD_DESCRIPTIONS[cloudClass.code] || { name: cloudClass.name };
        html += `
            <div class="cloud-type-badge">
                <div class="cloud-type-code">${cloudClass.code}</div>
                <div class="cloud-type-name">${info.name || cloudClass.name}</div>
            </div>
        `;
    });

    cloudTypesGrid.innerHTML = html;
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadCloudTypes();

    // Prevent file input from reopening on drag
    document.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
    });

    document.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
    });
});

// Keyboard Support
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && resultsSection.style.display === 'block') {
        resetUpload();
    }
});
