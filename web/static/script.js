/* ============================================================
   CloudClassify13 - JavaScript
   ============================================================ */

// Cloud type descriptions
const CLOUD_DESCRIPTIONS = {
    'Ci': {
        name: 'Cirrus',
        description: 'Delgadas y transparentes; blancas y brillantes; las nubes son filamentosas y parecidas a colas de caballo.'
    },
    'Cc': {
        name: 'Cirrocumulus',
        description: 'Nubes muy pequeñas, blancas y brillantes; son escamas blancas delgadas; a menudo dispuestas en filas y en grupos.'
    },
    'Cs': {
        name: 'Cirrostratus',
        description: 'La parte inferior de la nube tiene una estructura filamentosa; el cuerpo de la nube es lo suficientemente delgado para dejar pasar el sol y la luna; y hay un halo distinto bajo la iluminación del sol.'
    },
    'Ac': {
        name: 'Altocumulus',
        description: 'Nubes pequeñas y distintas en contorno; las nubes delgadas son blancas, contorno Sol-Luna visible, las nubes gruesas son gris oscuro, el contorno del Sol-Luna no es claro; nubes ovaladas, en forma de teja, escamas de pescado o distribución ondulada de agua.'
    },
    'As': {
        name: 'Altostratus',
        description: 'Nubes más gruesas y que cubren el cielo; el sol pasa a través casi sin halo; a menudo tienen estructuras rayadas y son gris-blanquecinas o gris-azuladas.'
    },
    'Cu': {
        name: 'Cumulus',
        description: 'Proyección hacia arriba de un arco circular; nubes de tamaño similar a puños; márgenes claros.'
    },
    'Cb': {
        name: 'Cumulonimbus',
        description: 'Nubes gruesas y con forma de brócoli; los bordes son borrosos.'
    },
    'Ns': {
        name: 'Nimbostratus',
        description: 'Nubes bajas y amorfas; a menudo cubren el cielo y oscurecen completamente el sol y la luna; nubes esponjosas y gris oscuro.'
    },
    'Sc': {
        name: 'Stratocumulus',
        description: 'Nubes generalmente del tamaño de un puño y distribuidas libremente, agrupadas, viajeras y onduladas, a menudo grises o gris-blanquecinas.'
    },
    'St': {
        name: 'Stratus',
        description: 'Nubes que yacen uniformemente; cubren una gran área, casi todo el cielo; mayormente grises.'
    },
    'Ct': {
        name: 'Contrails',
        description: 'Estelas de condensación; nubes en forma de línea producidas por el escape de los motores de los aviones.'
    },
    'No': {
        name: 'No Cloud',
        description: 'No hay nubes en el cielo.'
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
    const warning = data.warning;
    const isLikelyCloud = data.is_likely_cloud !== false;

    const cloudInfo = CLOUD_DESCRIPTIONS[prediction] || { name: prediction, description: 'Descripción no disponible.' };
    
    let html = `
        <div class="results-grid">
            <!-- Imagen a la izquierda -->
            <div class="image-column">
                ${selectedImageDataURL ? `
                <img src="${selectedImageDataURL}" alt="Imagen analizada" class="analyzed-image">
                ` : ''}
            </div>
            
            <!-- Resultados a la derecha -->
            <div class="results-column">
                ${!isLikelyCloud ? `
                <div class="warning-box">
                    <i class="fas fa-exclamation-triangle"></i>
                    <div>
                        <strong>Advertencia:</strong>
                        <p>${warning || 'La confianza es baja. Asegúrate de que la imagen contenga nubes visibles.'}</p>
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
            </div>
        </div>
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
            <div class="cloud-type-badge" id="badge-${cloudClass.code}" onclick="toggleCloudDescription('${cloudClass.code}')">
                <div class="badge-header">
                    <div class="header-content">
                        <div class="cloud-type-code">${cloudClass.code}</div>
                        <div class="cloud-type-name">${info.name || cloudClass.name}</div>
                    </div>
                    <div class="toggle-icon">
                        <i class="fas fa-chevron-down"></i>
                    </div>
                </div>
                <div class="cloud-type-description">
                    <p>${info.description || 'Sin descripción disponible.'}</p>
                </div>
            </div>
        `;
    });

    cloudTypesGrid.innerHTML = html;
}

function toggleCloudDescription(code) {
    // Toggle current only
    const badge = document.getElementById(`badge-${code}`);
    if (badge) {
        badge.classList.toggle('active');
    }
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
