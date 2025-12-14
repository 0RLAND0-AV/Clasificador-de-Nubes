"""
CloudClassify13 - Configuración Global del Proyecto
====================================================
Parámetros de hiperparámetros, rutas y configuración del proyecto.

Archivo: config.py
Descripción: Centraliza todos los parámetros de configuración del modelo,
             dataset y entrenamiento para fácil modificación.

================================================================================
GUÍA DE HIPERPARÁMETROS CNN - LEE ESTO ANTES DE CAMBIAR VALORES
================================================================================

Cada parámetro tiene documentación que explica:
  1. ¿Qué es y para qué sirve?
  2. ¿Qué pasa si lo AUMENTO?
  3. ¿Qué pasa si lo DISMINUYO?
  4. Valor recomendado

================================================================================
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
    'Ct': 'Contrails',        # Contrails - Estelas de aviones
    'Nc': 'Sin Nubes'         # No Cloud - Cielo despejado o sin nubes
}

NUM_CLASSES = len(CLOUD_CLASSES)
CLASS_NAMES = list(CLOUD_CLASSES.keys())

# ==================== PARÁMETROS DEL DATASET ====================

# ------------------------------------------------------------------------------
# IMAGE_SIZE (Tamaño de imagen de entrada)
# ------------------------------------------------------------------------------
# ¿Qué es?
#   Dimensión en píxeles a la que se redimensionan TODAS las imágenes.
#   Las imágenes se convierten a IMAGE_SIZE x IMAGE_SIZE antes de entrar a la CNN.
#
# ¿Qué pasa si lo AUMENTO? (ej: 224 → 448)
#   ✅ Más detalle: La CNN ve más detalles finos de las nubes
#   ✅ Mejor precisión potencial: Distingue patrones pequeños
#   ❌ 4x más lento: El entrenamiento tarda mucho más
#   ❌ Más memoria GPU: Puede dar "CUDA out of memory"
#   ❌ Overfitting: Con pocas imágenes, puede memorizar ruido
#
# ¿Qué pasa si lo DISMINUYO? (ej: 224 → 112)
#   ✅ 4x más rápido: Entrenamiento veloz
#   ✅ Menos memoria GPU: Funciona en GPUs pequeñas
#   ❌ Pierde detalle: Patrones finos se pierden
#   ❌ Peor precisión: Clases similares se confunden
#
# Valores típicos: 64, 112, 224, 299, 448
# Recomendado: 224 (balance entre calidad y velocidad)
# ------------------------------------------------------------------------------
IMAGE_SIZE = 224
IMAGE_HEIGHT = IMAGE_SIZE
IMAGE_WIDTH = IMAGE_SIZE
INPUT_CHANNELS = 3  # 3=RGB color, 1=escala de grises

# ------------------------------------------------------------------------------
# TRAIN_SPLIT / VAL_SPLIT / TEST_SPLIT (División de datos)
# ------------------------------------------------------------------------------
# ¿Qué es?
#   Porcentaje de imágenes para cada conjunto. DEBEN SUMAR 1.0
#   - TRAIN: Imágenes que el modelo USA para aprender
#   - VAL: Imágenes para medir progreso (NO las aprende)
#   - TEST: Imágenes para evaluación final
#
# ¿Qué pasa si AUMENTO TRAIN_SPLIT? (ej: 0.70 → 0.95)
#   ✅ El modelo ve más imágenes, aprende/memoriza mejor
#   ❌ Menos validación, no sabes si memoriza o generaliza
#
# ¿Qué pasa si DISMINUYO TRAIN_SPLIT? (ej: 0.70 → 0.50)
#   ❌ El modelo ve menos imágenes, aprende peor
#   ✅ Mejor validación y métricas más confiables
#
# Valores típicos:
#   Generalización: TRAIN=0.70, VAL=0.15, TEST=0.15
#   Memorización:   TRAIN=0.95, VAL=0.05, TEST=0.00
# ------------------------------------------------------------------------------
TRAIN_SPLIT = 0.95  # 95% para entrenamiento (modo memorización)
VAL_SPLIT = 0.05    # 5% para validación
TEST_SPLIT = 0.0    # 0% para test

# ==============================================================================
# ARQUITECTURA DE LA CNN - Capas Convolucionales y FC
# ==============================================================================

# ------------------------------------------------------------------------------
# CONV_LAYERS (Capas Convolucionales - Extracción de características)
# ------------------------------------------------------------------------------
# ¿Qué es?
#   Define los FILTROS que extraen características de las imágenes.
#   Cada tupla: (canales_entrada, num_filtros, kernel, stride, padding)
#   
#   - num_filtros: Cantidad de detectores de patrones (bordes, texturas, formas)
#   - kernel: Tamaño del filtro (3x3 es el estándar)
#   - Las primeras capas detectan bordes, las últimas detectan conceptos abstractos
#
# ¿Qué pasa si AUMENTO los filtros? (ej: 64→128→256→512 a 128→256→512→1024)
#   ✅ Más capacidad: Aprende patrones más complejos
#   ✅ Mejor para datasets grandes
#   ❌ Modelo más pesado (MB) y más lento
#   ❌ Más memoria GPU necesaria
#   ❌ Overfitting con pocas imágenes
#
# ¿Qué pasa si DISMINUYO los filtros? (ej: 64→128 a 16→32→64→128)
#   ✅ Modelo ligero y rápido
#   ✅ Menos overfitting con datasets pequeños
#   ❌ Menos capacidad de aprendizaje
#
# Regla general:
#   - <500 imágenes: Pocos filtros (16→32→64→128)
#   - 500-5000 imágenes: Filtros medios (32→64→128→256)
#   - >5000 imágenes: Muchos filtros (64→128→256→512)
# ------------------------------------------------------------------------------
CONV_LAYERS = [
    (3, 64, 3, 1, 1),      # Conv1: RGB(3) → 64 filtros (detecta bordes)
    (64, 128, 3, 1, 1),    # Conv2: 64 → 128 filtros (detecta texturas)
    (128, 256, 3, 1, 1),   # Conv3: 128 → 256 filtros (detecta formas)
    (256, 512, 3, 1, 1),   # Conv4: 256 → 512 filtros (detecta conceptos)
]

# ------------------------------------------------------------------------------
# FC_LAYERS (Capas Fully Connected - Clasificación)
# ------------------------------------------------------------------------------
# ¿Qué es?
#   Neuronas que DECIDEN la clase basándose en las características extraídas.
#   La última capa siempre tiene NUM_CLASSES neuronas (una por tipo de nube).
#
# ¿Qué pasa si AUMENTO neuronas?
#   ✅ Más capacidad de decisión
#   ❌ Más parámetros, modelo más pesado
#   ❌ Riesgo de overfitting
#
# ¿Qué pasa si DISMINUYO neuronas?
#   ✅ Modelo más ligero
#   ❌ Puede no capturar relaciones complejas
# ------------------------------------------------------------------------------
FC_LAYERS = [512, 256, 128]

# ------------------------------------------------------------------------------
# DROPOUT_RATE (Regularización)
# ------------------------------------------------------------------------------
# ¿Qué es?
#   Probabilidad de "apagar" neuronas aleatoriamente durante entrenamiento.
#   Previene que el modelo memorice (overfitting).
#   0.0 = sin dropout (memoriza todo), 0.5 = 50% neuronas apagadas
#
# ¿Qué pasa si lo AUMENTO? (ej: 0.3 → 0.7)
#   ✅ Menos overfitting, mejor generalización
#   ❌ Underfitting si es muy alto (no aprende nada)
#   ❌ Entrenamiento más lento
#
# ¿Qué pasa si lo DISMINUYO? (ej: 0.5 → 0.0)
#   ✅ Aprende/memoriza más rápido
#   ❌ Más overfitting
#
# Valores típicos:
#   Generalizar: 0.3 a 0.5
#   Memorizar: 0.0 a 0.1
# ------------------------------------------------------------------------------
DROPOUT_RATE = 0.0  # Sin dropout = modo memorización

# ==============================================================================
# HIPERPARÁMETROS DE ENTRENAMIENTO - Los más importantes
# ==============================================================================

# ------------------------------------------------------------------------------
# BATCH_SIZE (Tamaño del lote)
# ------------------------------------------------------------------------------
# ¿Qué es?
#   Número de imágenes procesadas JUNTAS antes de actualizar pesos.
#   La GPU procesa BATCH_SIZE imágenes en paralelo.
#
# ¿Qué pasa si lo AUMENTO? (ej: 8 → 32)
#   ✅ Más rápido: Aprovecha mejor el paralelismo de la GPU
#   ✅ Gradientes más estables
#   ❌ Más memoria GPU: Puede dar "CUDA out of memory"
#   ❌ Generalización peor en algunos casos
#
# ¿Qué pasa si lo DISMINUYO? (ej: 32 → 4)
#   ✅ Menos memoria GPU
#   ✅ Puede generalizar mejor (más ruido = explora más)
#   ❌ Más lento
#   ❌ Gradientes más ruidosos
#
# Valores según GPU:
#   GPU 4GB: batch 4-8
#   GPU 8GB: batch 16-32
#   GPU 16GB+: batch 32-64
# ------------------------------------------------------------------------------
BATCH_SIZE = 8

# ------------------------------------------------------------------------------
# LEARNING_RATE (Tasa de aprendizaje) ⭐ EL MÁS IMPORTANTE ⭐
# ------------------------------------------------------------------------------
# ¿Qué es?
#   Qué tanto se ajustan los pesos en cada paso.
#   Controla la "velocidad" de aprendizaje. ES EL HIPERPARÁMETRO MÁS CRÍTICO.
#
# ¿Qué pasa si lo AUMENTO? (ej: 0.001 → 0.01)
#   ✅ Aprende más rápido
#   ❌ Inestabilidad: Loss sube y baja caóticamente
#   ❌ Puede "explotar" (loss = inf o NaN)
#   ❌ Salta sobre el punto óptimo
#
# ¿Qué pasa si lo DISMINUYO? (ej: 0.001 → 0.0001)
#   ✅ Más estable, convergencia suave
#   ✅ Encuentra mejores mínimos
#   ❌ MUY lento, necesita muchas épocas
#   ❌ Puede quedarse atascado
#
# Valores típicos:
#   Adam: 0.0001 a 0.001
#   SGD: 0.01 a 0.1
# ------------------------------------------------------------------------------
LEARNING_RATE = 0.001

# ------------------------------------------------------------------------------
# WEIGHT_DECAY (Regularización L2)
# ------------------------------------------------------------------------------
# ¿Qué es?
#   Penaliza pesos grandes para prevenir overfitting.
#   loss_total = loss + weight_decay * sum(pesos²)
#
# ¿Qué pasa si lo AUMENTO? (ej: 0 → 0.01)
#   ✅ Menos overfitting
#   ❌ Puede causar underfitting si es muy alto
#
# ¿Qué pasa si lo DISMINUYO? (ej: 0.001 → 0)
#   ✅ Pesos pueden crecer libremente
#   ❌ Más riesgo de overfitting
#
# Para memorización: 0.0 (sin restricción)
# Para generalización: 0.0001 a 0.01
# ------------------------------------------------------------------------------
WEIGHT_DECAY = 0.0

# ------------------------------------------------------------------------------
# EPOCHS (Número de épocas)
# ------------------------------------------------------------------------------
# ¿Qué es?
#   Cuántas veces el modelo ve TODO el dataset completo.
#   1 época = 1 pasada por todas las imágenes.
#
# ¿Qué pasa si lo AUMENTO? (ej: 50 → 200)
#   ✅ Más oportunidad de aprender/memorizar
#   ❌ Más tiempo de entrenamiento
#   ❌ Overfitting si no hay early stopping
#
# ¿Qué pasa si lo DISMINUYO? (ej: 100 → 20)
#   ✅ Entrenamiento rápido
#   ❌ El modelo no alcanza a converger
#   ❌ Accuracy baja
#
# Valores típicos: 50 a 300
# ------------------------------------------------------------------------------
EPOCHS = 200

# ------------------------------------------------------------------------------
# OPTIMIZER_TYPE (Algoritmo de optimización)
# ------------------------------------------------------------------------------
# ¿Qué es?
#   Algoritmo que actualiza los pesos del modelo.
#
#   'adam': Adapta LR por parámetro. Rápido y popular. Recomendado para empezar.
#   'sgd': Gradiente descendente clásico. Más lento pero mejor generalización.
#   'rmsprop': Similar a Adam. Bueno para RNNs.
# ------------------------------------------------------------------------------
OPTIMIZER_TYPE = 'adam'
OPTIMIZER = 'adam'

# ------------------------------------------------------------------------------
# LOSS_FUNCTION (Función de pérdida)
# ------------------------------------------------------------------------------
# ¿Qué es?
#   Mide el error entre predicción y etiqueta real.
#   El modelo MINIMIZA esta función.
#
#   'crossentropy': Estándar para clasificación. Funciona para la mayoría.
#   'focal_loss': Da más peso a ejemplos difíciles (clases desbalanceadas).
# ------------------------------------------------------------------------------
LOSS_FUNCTION = 'crossentropy'

# ------------------------------------------------------------------------------
# EARLY_STOPPING (Detención temprana)
# ------------------------------------------------------------------------------
# ¿Qué es?
#   Si val_loss no mejora en PATIENCE épocas, PARA el entrenamiento.
#   Previene overfitting y ahorra tiempo.
#
# PATIENCE alto (100+): Entrena hasta el final (modo memorización)
# PATIENCE bajo (10-20): Para cuando deja de mejorar
# ------------------------------------------------------------------------------
EARLY_STOPPING_PATIENCE = 100
EARLY_STOPPING_MIN_DELTA = 0.001

# ------------------------------------------------------------------------------
# LR_SCHEDULER (Reducción del Learning Rate)
# ------------------------------------------------------------------------------
# ¿Qué es?
#   Reduce el LR durante el entrenamiento.
#   Empieza alto (aprende rápido) → termina bajo (afina detalles)
#
#   'cosine': Curva coseno suave (recomendado)
#   'step': Reduce a la mitad cada N épocas
#   'exponential': Reduce multiplicando por factor
#
# T_MAX: Épocas para un ciclo completo del coseno
# ------------------------------------------------------------------------------
USE_LR_SCHEDULER = True
LR_SCHEDULER_TYPE = 'cosine'
SCHEDULER = 'cosine'
LR_SCHEDULER_T_MAX = 30

# ==============================================================================
# DATA AUGMENTATION (Aumento de Datos) - Técnicas para enriquecer el dataset
# ==============================================================================
# ¿Qué es Data Augmentation?
#   Transforma las imágenes durante el entrenamiento (rotar, voltear, etc.)
#   para crear "variaciones" artificiales. Así el modelo ve más ejemplos.
#
# ¿Cuándo DESACTIVAR? (USE_DATA_AUGMENTATION = False)
#   - Cuando quieres MEMORIZAR el dataset exacto
#   - Dataset muy pequeño donde cada imagen es única e importante
#   - Cuando la orientación importa (ej: suelo siempre abajo)
#
# ¿Cuándo ACTIVAR? (USE_DATA_AUGMENTATION = True)
#   - Cuando quieres GENERALIZAR a nuevas imágenes
#   - Dataset grande donde las transformaciones agregan variedad útil
# ------------------------------------------------------------------------------
USE_DATA_AUGMENTATION = True

# ------------------------------------------------------------------------------
# AUGMENTATION_PARAMS - Parámetros específicos de cada transformación
# ------------------------------------------------------------------------------
#
# random_horizontal_flip: Voltear horizontalmente (espejo)
#   ✅ Generalmente seguro para nubes
#   Las nubes se ven igual volteadas horizontalmente
#
# random_vertical_flip: Voltear verticalmente
#   ⚠️ PRECAUCIÓN: El suelo/cielo se invierte
#   Normalmente DESACTIVADO porque el contexto del suelo ayuda
#
# random_rotation: Rotar la imagen N grados
#   Valor = ángulo máximo de rotación
#   10 = rota entre -10° y +10° (sutil)
#   45 = rota entre -45° y +45° (agresivo)
#   ⚠️ Mucha rotación puede deformar las nubes
#
# random_crop: Recortar un área aleatoria
#   ⚠️ PELIGROSO: Puede recortar la nube principal
#   Para memorización: DESACTIVADO
#
# random_brightness: Cambiar brillo (0.0 a 1.0)
#   0.1 = cambio muy leve, 0.5 = cambio drástico
#   Simula diferentes condiciones de luz
#
# random_contrast: Cambiar contraste (0.0 a 1.0)
#   Simula cámaras con diferentes configuraciones
#
# random_saturation: Cambiar saturación de colores (0.0 a 1.0)
#   0.0 = sin cambio, 1.0 = muy saturado o desaturado
#
# random_hue: Cambiar tono de color (0.0 a 0.5)
#   ⚠️ PELIGROSO: Puede cambiar el color del cielo
#   Para nubes: DESACTIVADO (0.0)
#
# gaussian_blur: Aplicar desenfoque
#   Simula fotos borrosas o con poca nitidez
# ------------------------------------------------------------------------------
AUGMENTATION_PARAMS = {
    'random_horizontal_flip': True,
    'random_vertical_flip': False,   # Desactivado: El suelo ayuda a identificar
    'random_rotation': 10,           # Reducido: Rotar mucho deforma la nube
    'random_crop': False,            # DESACTIVADO: Importante ver foto completa
    'random_brightness': 0.1,        # Muy leve
    'random_contrast': 0.1,          # Muy leve
    'random_saturation': 0.1,        # Muy leve
    'random_hue': 0.0,               # Desactivado: No cambiar colores reales
    'gaussian_blur': False,
    'gaussian_blur_kernel': 3,
}

# ------------------------------------------------------------------------------
# NORMALIZACIÓN (Estadísticas de ImageNet)
# ------------------------------------------------------------------------------
# ¿Qué es?
#   Ajusta los valores de píxeles para que tengan media~0 y std~1.
#   Los modelos preentrenados esperan estos valores específicos.
#
# ¿Por qué estos valores específicos (0.485, 0.456, 0.406)?
#   Son las estadísticas calculadas de millones de imágenes de ImageNet.
#   Si usas transfer learning de ImageNet, USA ESTOS VALORES.
#
# ¿Cuándo cambiarlos?
#   - Si entrenas desde cero en un dominio MUY diferente
#   - Si tus imágenes tienen distribución de colores muy distinta
# ------------------------------------------------------------------------------
NORMALIZE_MEAN = [0.485, 0.456, 0.406]  # Media RGB de ImageNet
NORMALIZE_STD = [0.229, 0.224, 0.225]   # Desviación estándar RGB de ImageNet

# ==============================================================================
# DISPOSITIVO DE COMPUTACIÓN (CPU vs GPU)
# ==============================================================================
# ¿Qué es USE_CUDA?
#   True = Usa GPU (NVIDIA) si está disponible
#   False = Fuerza uso de CPU
#
# GPU vs CPU:
#   GPU: 10x a 100x más rápido en entrenamiento
#   CPU: Más lento pero siempre disponible
#
# Si tienes GPU pero da errores de memoria:
#   1. Reduce BATCH_SIZE
#   2. Reduce IMAGE_SIZE
#   3. Si sigue fallando, usa CPU (USE_CUDA = False)
# ------------------------------------------------------------------------------
USE_CUDA = True

# ==============================================================================
# PARÁMETROS DE LOGGING Y GUARDADO
# ==============================================================================
# Crear directorios si no existen
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(WEB_DIR, exist_ok=True)
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

# LOG_FREQUENCY: Cada cuántos batches mostrar información en consola
#   Valor bajo (5): Mucha información, útil para debug
#   Valor alto (50): Menos spam en consola
LOG_FREQUENCY = 10

# CHECKPOINT_FREQUENCY: Cada cuántas épocas guardar el modelo
#   Valor bajo (1): Guarda cada época (usa más espacio en disco)
#   Valor alto (10): Guarda menos frecuentemente
CHECKPOINT_FREQUENCY = 5

# ==============================================================================
# PARÁMETROS DE PREDICCIÓN (para el clasificador web)
# ==============================================================================

# MIN_CONFIDENCE: Confianza mínima para aceptar una predicción
#   0.0 = Siempre predice algo (recomendado para memorización)
#   0.3 = Rechaza predicciones con <30% confianza
#   0.5 = Solo acepta predicciones con >50% confianza
#
#   ⚠️ Con 0.0, NUNCA dirá "Unknown", siempre predice un tipo de nube
MIN_CONFIDENCE = 0.0

# TOP_K: Cuántas predicciones alternativas mostrar
#   1 = Solo la predicción principal
#   3 = Muestra las 3 más probables
#   5 = Muestra las 5 más probables (útil para análisis)
TOP_K = 3

# ==============================================================================
# PARÁMETROS DE LA APLICACIÓN WEB (Flask)
# ==============================================================================

# Puerto del servidor web
#   5000 = Puerto por defecto de Flask
#   80 = Puerto HTTP estándar (requiere permisos de administrador)
WEB_PORT = 5000

# Tamaño máximo de archivo a subir (en bytes)
#   5 * 1024 * 1024 = 5 MB
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB

# Extensiones de imagen permitidas
#   Todas las extensiones que el modelo puede procesar
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'avif', 'tiff', 'tif'}

# ==============================================================================
# INFORMACIÓN DEL PROYECTO
# ==============================================================================
PROJECT_NAME = "CloudClassify13"
PROJECT_VERSION = "1.0.0"
PROJECT_DESCRIPTION = "Clasificación automática de tipos de nubes usando CNN"
PROJECT_AUTHOR = "Grupo #13 - Inteligencia Artificial"

# ==============================================================================
# RESUMEN DE CONFIGURACIÓN AL INICIAR
# ==============================================================================
print(f"[OK] Configuracion cargada: {PROJECT_NAME} v{PROJECT_VERSION}")
print(f"[OK] Clases a clasificar: {NUM_CLASSES}")
print(f"[OK] Tamano de imagen: {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"[OK] GPU disponible: {'Si (CUDA)' if USE_CUDA else 'No'}")
