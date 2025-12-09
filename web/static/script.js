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

// Global variables
let uploadArea, fileInput, uploadSection, resultsSection, resultContent;
let selectedFile = null;
let selectedImageDataURL = null; // Para guardar la imagen

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    uploadArea = document.getElementById('upload-area');
    fileInput = document.getElementById('file-input');
    uploadSection = document.getElementById('upload-section');
    resultsSection = document.getElementById('results-section');
    resultContent = document.getElementById('result-content');

    // Setup event listeners
    setupEventListeners();
    
    // Load cloud types
    loadCloudTypes();
    loadProjectInfo();
});

function setupEventListeners() {
    // Click to upload
    uploadArea.addEventListener('click', function(e) {
        e.preventDefault();
        e.stopPropagation();
        fileInput.click();
    });

    // File input change
    fileInput.addEventListener('change', function(e) {
        e.preventDefault();
        e.stopPropagation();
        if (e.target.files && e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    // Drag and drop
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.add('drag-over');
    });

    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.remove('drag-over');
    });

    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.remove('drag-over');
        
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    // Prevent default drag/drop on document
    document.addEventListener('dragover', function(e) {
        e.preventDefault();
    });

    document.addEventListener('drop', function(e) {
        e.preventDefault();
    });
}

// Handle file selection
function handleFileSelect(file) {
    // Validate file
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp', 'image/gif', 'image/bmp'];
    const maxSize = 10 * 1024 * 1024; // 10MB

    if (!validTypes.includes(file.type.toLowerCase())) {
        showError('Formato de archivo no válido. Use JPG, PNG, WebP, GIF o BMP.');
        return;
    }

    if (file.size > maxSize) {
        showError('El archivo es muy grande. Máximo 10MB.');
        return;
    }

    selectedFile = file;

    // Show preview
    showImagePreview(file);

    // Upload automatically
    uploadImage(file);
}

// Show image preview
function showImagePreview(file) {
    const reader = new FileReader();
    
    reader.onload = function(e) {
        selectedImageDataURL = e.target.result; // Guardar para mostrar después
        uploadArea.innerHTML = `
            <div class="image-preview" style="text-align: center;">
                <img src="${e.target.result}" alt="Preview" style="max-width: 100%; max-height: 300px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <p style="margin-top: 10px; font-size: 14px; color: #666; font-weight: 500;">${file.name}</p>
            </div>
        `;
    };
    
    reader.readAsDataURL(file);
}

// Upload image
function uploadImage(file) {
    showLoading();

    const formData = new FormData();
    formData.append('file', file);

    fetch('/api/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(data => {
                throw new Error(data.error || `Error HTTP: ${response.status}`);
            });
        }
        return response.json();
    })
    .then(data => {
        if (data.success) {
            displayResults(data);
        } else {
            showError(data.error || 'Error desconocido en la predicción');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showError(`Error al conectar con el servidor: ${error.message}`);
    });
}

// Display Results
function displayResults(data) {
    const prediction = data.predicted_class;
    const confidence = data.confidence;
    const top_k = data.top_predictions || [];
    const warning = data.warning;  // Advertencia si no es nube
    const isLikelyCloud = data.is_likely_cloud !== false;  // Por defecto true

    // Main prediction
    const cloudInfo = CLOUD_DESCRIPTIONS[prediction] || { name: prediction, description: 'Descripción no disponible.' };
    
    let html = `
        <!-- Imagen subida -->
        ${selectedImageDataURL ? `
        <div style="text-align: center; margin-bottom: 20px;">
            <img src="${selectedImageDataURL}" alt="Imagen analizada" style="max-width: 100%; max-height: 400px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
        </div>
        ` : ''}
        
        <!-- Advertencia si no es nube -->
        ${!isLikelyCloud ? `
        <div style="background: #fef3c7; border-left: 4px solid #f59e0b; padding: 15px; margin-bottom: 20px; border-radius: 8px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <i class="fas fa-exclamation-triangle" style="color: #f59e0b; font-size: 1.5em;"></i>
                <div>
                    <strong style="color: #92400e;">Advertencia:</strong>
                    <p style="margin: 5px 0 0 0; color: #78350f;">${warning || 'La confianza es baja. Asegúrate de que la imagen contenga nubes visibles.'}</p>
                </div>
            </div>
        </div>
        ` : ''}

        <div class="result-main">
            <div class="result-class">
                <div class="label">Tipo de Nube Identificada</div>
                <h3 style="color: #1e40af; font-size: 2.5em; margin: 10px 0;">${prediction}</h3>
                <div class="class-code" style="font-size: 1.2em; color: #666;">${cloudInfo.name}</div>
            </div>

            <div class="confidence-section" style="margin: 20px 0;">
                <div class="label">Nivel de Confianza</div>
                <div class="confidence-bar" style="background: #e5e7eb; height: 30px; border-radius: 15px; overflow: hidden; margin: 10px 0;">
                    <div class="confidence-fill" style="background: linear-gradient(90deg, #10b981, #059669); height: 100%; width: 0%; transition: width 1s ease;"></div>
                </div>
                <div class="confidence-text" style="font-size: 1.5em; font-weight: bold; color: #059669;">${(confidence * 100).toFixed(1)}%</div>
            </div>
        </div>

        <div class="result-description" style="background: #f0f9ff; padding: 15px; border-radius: 8px; margin: 20px 0;">
            <h4><i class="fas fa-cloud"></i> Descripción</h4>
            <p style="line-height: 1.6;">${cloudInfo.description}</p>
        </div>

        <div class="result-detailed">
            <h4><i class="fas fa-list"></i> Top 3 Predicciones</h4>
            <div class="top-predictions">
    `;

    top_k.forEach((item, index) => {
        const itemCode = item.class;
        const itemConf = item.probability;
        const itemName = CLOUD_DESCRIPTIONS[itemCode]?.name || itemCode;
        const bgColor = index === 0 ? '#e0f2fe' : '#f9fafb';
        
        html += `
            <div class="prediction-item" style="background: ${bgColor}; padding: 12px; margin: 8px 0; border-radius: 8px; border-left: 4px solid ${index === 0 ? '#0284c7' : '#cbd5e1'};">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                    <div>
                        <span style="font-weight: bold; font-size: 1.1em;">${itemCode}</span>
                        <span style="color: #666; margin-left: 8px;">${itemName}</span>
                    </div>
                    <span style="font-weight: bold; color: #059669;">${(itemConf * 100).toFixed(1)}%</span>
                </div>
                <div class="prediction-bar" style="background: #e5e7eb; height: 8px; border-radius: 4px; overflow: hidden;">
                    <div class="prediction-bar-fill" style="background: #10b981; height: 100%; width: ${itemConf * 100}%; transition: width 0.5s ease;"></div>
                </div>
            </div>
        `;
    });

    html += `
            </div>
        </div>

        <button class="btn btn-secondary" onclick="resetUpload()" style="margin-top: 20px; width: 100%; padding: 12px; background: #6366f1; color: white; border: none; border-radius: 8px; cursor: pointer; font-size: 16px;">
            <i class="fas fa-redo"></i> Clasificar otra imagen
        </button>
    `;

    resultContent.innerHTML = html;
    resultsSection.style.display = 'block';
    uploadSection.style.display = 'none';
    
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
    resultContent.innerHTML = `
        <div class="alert alert-error" style="background: #fee; border: 2px solid #f87171; padding: 20px; border-radius: 8px; text-align: center;">
            <i class="fas fa-exclamation-circle" style="font-size: 3em; color: #dc2626; margin-bottom: 10px;"></i>
            <p style="font-size: 1.1em; color: #991b1b; margin: 10px 0;">${message}</p>
        </div>
        <button class="btn btn-secondary" onclick="resetUpload()" style="margin-top: 20px; width: 100%; padding: 12px; background: #6366f1; color: white; border: none; border-radius: 8px; cursor: pointer; font-size: 16px;">
            <i class="fas fa-redo"></i> Intentar de nuevo
        </button>
    `;
    resultsSection.style.display = 'block';
    uploadSection.style.display = 'none';
}

// Show Loading
function showLoading() {
    resultContent.innerHTML = `
        <div class="loading" style="text-align: center; padding: 40px;">
            <div class="spinner" style="border: 4px solid #f3f3f3; border-top: 4px solid #3498db; border-radius: 50%; width: 50px; height: 50px; animation: spin 1s linear infinite; margin: 0 auto;"></div>
            <p style="margin-top: 20px; font-size: 1.2em; color: #666;">Analizando imagen...</p>
            <p style="color: #999;">Esto puede tomar unos segundos</p>
        </div>
        <style>
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    `;
    resultsSection.style.display = 'block';
    uploadSection.style.display = 'none';
}

// Reset Upload
function resetUpload() {
    uploadSection.style.display = 'block';
    resultsSection.style.display = 'none';
    resultContent.innerHTML = '';
    selectedFile = null;
    selectedImageDataURL = null; // Limpiar la imagen guardada
    
    // Reset file input
    fileInput.value = '';
    
    // Reset upload area
    uploadArea.innerHTML = `
        <i class="fas fa-image" style="font-size: 4em; color: #94a3b8; margin-bottom: 15px;"></i>
        <p><strong>Arrastra una imagen aquí</strong></p>
        <p class="small">o haz clic para seleccionar</p>
    `;
    
    uploadArea.classList.remove('drag-over');
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
    const cloudTypesGrid = document.getElementById('cloudTypesList');
    if (!cloudTypesGrid) return;

    let html = '';
    classes.forEach(cloudClass => {
        const info = CLOUD_DESCRIPTIONS[cloudClass.code] || { name: cloudClass.name };
        html += `
            <div class="cloud-type-badge" style="background: #f0f9ff; padding: 10px; margin: 5px; border-radius: 8px; text-align: center; border: 2px solid #bae6fd;">
                <div class="cloud-type-code" style="font-weight: bold; font-size: 1.2em; color: #0284c7;">${cloudClass.code}</div>
                <div class="cloud-type-name" style="font-size: 0.9em; color: #666;">${info.name || cloudClass.name}</div>
            </div>
        `;
    });

    cloudTypesGrid.innerHTML = html;
}

// Load Project Info
async function loadProjectInfo() {
    try {
        const response = await fetch('/api/info');
        if (!response.ok) throw new Error('Failed to load project info');
        
        const data = await response.json();
        
        // Update info sections
        const numClassesEl = document.getElementById('numClasses');
        const deviceInfoEl = document.getElementById('deviceInfo');
        const imageSizeEl = document.getElementById('imageSize');
        
        if (numClassesEl) numClassesEl.textContent = `${data.num_classes} tipos`;
        if (deviceInfoEl) deviceInfoEl.textContent = data.device;
        if (imageSizeEl) imageSizeEl.textContent = `${data.image_size}×${data.image_size}px`;
    } catch (error) {
        console.error('Error loading project info:', error);
    }
}

// Keyboard Support
document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape' && resultsSection && resultsSection.style.display === 'block') {
        resetUpload();
    }
});
