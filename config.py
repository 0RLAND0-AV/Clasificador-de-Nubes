"""
CloudClassify13 - Configuración Global del Proyecto
====================================================
Parámetros de hiperparámetros, rutas y configuración del proyecto.

Archivo: config.py
Descripción: Centraliza todos los parámetros de configuración del modelo,
             dataset y entrenamiento para fácil modificación.
"""

import os
from pathlib import Path

# ==================== RUTAS ====================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
WEB_DIR = PROJECT_ROOT / "web"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Archivo del modelo entrenado
MODEL_PATH = MODELS_DIR / "cloud_classifier_best.pth"
MODEL_WEIGHTS_PATH = MODELS_DIR / "cloud_classifier_weights.pth"

# ==================== CLASES DE NUBES ====================
# Clasificación de 11 tipos de nubes según la OMM (Organización Meteorológica Mundial)
CLOUD_CLASSES = {
    'Ci': 'Cirros',           # Cirrus - Nubes altas, filamentosas
    'Cc': 'Cirrocúmulos',     # Cirrocumulus - Nubes altas, pequeñas masas
    'Cs': 'Cirrostratos',     # Cirrostratus - Nubes altas, velo transparente
    'Ac': 'Altocúmulos',      # Altocumulus - Nubes medias, masas blancas
    'As': 'Altostratos',      # Altostratus - Nubes medias, capa grisácea
    'Cu': 'Cúmulos',          # Cumulus - Nubes bajas, contornos definidos
    'Cb': 'Cumulonimbos',     # Cumulonimbus - Nubes de tormenta
    'Ns': 'Nimboestratos',    # Nimbostratus - Nubes oscuras de lluvia
    'Sc': 'Estratocúmulos',   # Stratocumulus - Nubes bajas, en capas
    'St': 'Estratos',         # Stratus - Nubes bajas, capa uniforme
    'Ct': 'Contrails'         # Contrails - Estelas de aviones
}

NUM_CLASSES = len(CLOUD_CLASSES)
CLASS_NAMES = list(CLOUD_CLASSES.keys())

# ==================== PARÁMETROS DEL DATASET ====================
# Tamaño de imagen de entrada
IMAGE_SIZE = 224
IMAGE_HEIGHT = IMAGE_SIZE
IMAGE_WIDTH = IMAGE_SIZE
INPUT_CHANNELS = 3

# Porcentaje de datos para entrenamiento, validación y test
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# ==================== PARÁMETROS DEL MODELO CNN ====================
# Arquitectura de capas convolucionales
# Formato: (in_channels, out_channels, kernel_size, stride, padding)
CONV_LAYERS = [
    (3, 64, 3, 1, 1),      # Conv1: 3->64 canales
    (64, 128, 3, 1, 1),    # Conv2: 64->128 canales
    (128, 256, 3, 1, 1),   # Conv3: 128->256 canales
    (256, 512, 3, 1, 1),   # Conv4: 256->512 canales
]

# Arquitectura de capas completamente conectadas
FC_LAYERS = [512, 256, 128]

# Dropout rate (regularización) - aumentado para evitar overfitting
DROPOUT_RATE = 0.6  # Aumentado de 0.5

# ==================== PARÁMETROS DE ENTRENAMIENTO ====================
# Hiperparámetros de optimización
BATCH_SIZE = 16          # Reducido de 32 (mejor para datasets pequeños)
LEARNING_RATE = 0.0005   # Reducido de 0.001 (aprendizaje más lento pero estable)
WEIGHT_DECAY = 1e-4
EPOCHS = 100

# Optimizer: 'adam', 'sgd', 'rmsprop'
OPTIMIZER_TYPE = 'adam'
OPTIMIZER = 'adam'  # Alias para compatibilidad

# Loss function: 'crossentropy', 'focal_loss'
LOSS_FUNCTION = 'crossentropy'

# Early stopping
EARLY_STOPPING_PATIENCE = 30  # Aumentado de 25 a 30
EARLY_STOPPING_MIN_DELTA = 0.001

# Learning rate scheduler
USE_LR_SCHEDULER = True
LR_SCHEDULER_TYPE = 'cosine'  # 'cosine', 'step', 'exponential'
SCHEDULER = 'cosine'  # Alias para compatibilidad
LR_SCHEDULER_T_MAX = 30  # Para CosineAnnealingLR

# ==================== DATA AUGMENTATION ====================
# Técnicas de aumento de datos (aplicadas EN TIEMPO REAL durante entrenamiento)
USE_DATA_AUGMENTATION = True
AUGMENTATION_PARAMS = {
    'random_horizontal_flip': True,
    'random_vertical_flip': False,     # Las nubes no se voltean verticalmente
    'random_rotation': 20,              # Aumentado de 15 a 20 grados
    'random_crop': False,               # Desactivado para no perder info
    'random_brightness': 0.15,          # Reducido para preservar características
    'random_contrast': 0.15,            # Reducido para preservar características
    'random_saturation': 0.0,           # Desactivado - color es importante
    'random_hue': 0.0,                  # Desactivado - tono es importante
    'gaussian_blur': False,
    'gaussian_blur_kernel': 5,
}

# Normalización (estadísticas de ImageNet)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# ==================== DISPOSITIVO DE COMPUTACIÓN ====================
# Usar GPU si está disponible
USE_CUDA = True

# ==================== PARÁMETROS DE LOGGING Y GUARDADO ====================
# Crear directorios si no existen
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(WEB_DIR, exist_ok=True)
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

# Frecuencia de logging (cada N batches)
LOG_FREQUENCY = 10

# Guardar checkpoint cada N épocas
CHECKPOINT_FREQUENCY = 5

# ==================== PARÁMETROS DE PREDICCIÓN ====================
# Confianza mínima para mostrar resultado
MIN_CONFIDENCE = 0.4  # Umbral más alto para mejor precisión

# Top-K predicciones a mostrar
TOP_K = 3

# Umbral para detectar "no es nube"
NO_CLOUD_THRESHOLD = 0.25  # Si la mejor predicción < 25%, probablemente no es nube

# ==================== PARÁMETROS DE LA WEB ====================
# Puerto del servidor web
WEB_PORT = 5000

# Tamaño máximo de archivo a subir
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB

# Extensiones permitidas
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'bmp'}

# ==================== INFORMACIÓN DEL PROYECTO ====================
PROJECT_NAME = "CloudClassify13"
PROJECT_VERSION = "1.0.0"
PROJECT_DESCRIPTION = "Clasificación automática de tipos de nubes usando CNN"
PROJECT_AUTHOR = "Grupo #13 - Inteligencia Artificial"

print(f"✓ Configuración cargada: {PROJECT_NAME} v{PROJECT_VERSION}")
print(f"✓ Clases a clasificar: {NUM_CLASSES}")
print(f"✓ Tamaño de imagen: {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"✓ Dispositivo: {'CUDA (GPU)' if USE_CUDA else 'CPU'}")
