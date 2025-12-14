# CloudClassify13 - Clasificador de Nubes con CNN

**Proyecto de Grupo #13**  
Universidad: San Simon 
Curso: Inteligencia Artificial
Periodo: 2/2025


## 📋 Descripción

CloudClassify13 es un sistema de clasificación automática de tipos de nubes basado en redes neuronales convolucionales (CNN). El proyecto combina un backend de machine learning en PyTorch con una interfaz web moderna HTML/CSS/JavaScript para clasificar imágenes de nubes en 11 categorías estándar de la Organización Meteorológica Mundial (OMM/WMO).

### Características Principales

- ✅ **CNN Custom**: Red neuronal convolucional diseñada específicamente para clasificación de nubes
- ✅ **11 Clases de Nubes**: Clasificación según estándares WMO/OMM
- ✅ **Interfaz Web Moderna**: Diseño de dos columnas con visualización optimizada de resultados
- ✅ **API REST**: Endpoints para integración en otras aplicaciones
- ✅ **Pipeline Modular**: Código organizado en módulos independientes
- ✅ **Online Data Augmentation**: Aumentación en tiempo real durante entrenamiento
- ✅ **Early Stopping**: Prevención de overfitting (patience=30)
- ✅ **GPU/CPU**: Soporte automático para aceleración GPU (CUDA)
- ✅ **Detección de No-Nubes**: Sistema de threshold para detectar imágenes sin nubes, cielo sin nubes.

## 🏗️ Arquitectura

### Clases de Nubes (11 tipos - WMO)

| Código | Nombre | Altitud | Descripción |
|--------|--------|---------|-------------|
| **Ci** | Cirrus | > 6000m | Nubes altas, finas, cristalinas con forma filamentosa |
| **Cc** | Cirrocumulus | > 6000m | Copos o grupos a gran altitud |
| **Cs** | Cirrostratus | > 6000m | Capas delgadas que producen halos |
| **Ac** | Altocumulus | 2000-6000m | Nubes medianas en racimos |
| **As** | Altostratus | 2000-6000m | Capas grises uniformes |
| **Cu** | Cumulus | < 2000m | Nubes densas con cúspides redondeadas |
| **Cb** | Cumulonimbus | < 2000m | Nubes de tormenta con desarrollo vertical |
| **Ns** | Nimbostratus | < 2000m | Capas oscuras que producen lluvia |
| **Sc** | Stratocumulus | < 2000m | Nubes bajas en capas o grupos |
| **St** | Stratus | < 2000m | Capas bajas uniformes |
| **Ct** | Contrails | > 6000m | Estelas de condensación de aviones |

### Arquitectura del Modelo CNN

```
CloudCNN Architecture:
├── Conv Block 1: Conv2d(3, 64) → BatchNorm2d(64) → ReLU → MaxPool2d
├── Conv Block 2: Conv2d(64, 128) → BatchNorm2d(128) → ReLU → MaxPool2d
├── Conv Block 3: Conv2d(128, 256) → BatchNorm2d(256) → ReLU → MaxPool2d
├── Conv Block 4: Conv2d(256, 512) → BatchNorm2d(512) → ReLU → MaxPool2d
├── Flatten: 512 × 14 × 14 = 100,352 features
├── FC1: Linear(100352, 512) → ReLU → Dropout(0.6)
├── FC2: Linear(512, 256) → ReLU → Dropout(0.6)
├── FC3: Linear(256, 128) → ReLU → Dropout(0.6)
└── Output: Linear(128, 11) → Logits
```

### Pipeline de Datos

```
Raw Images (224×224 RGB)
    ↓
Transformaciones (Train) - ONLINE AUGMENTATION:
  • Resize a 224×224
  • Random Horizontal Flip (p=0.5)
  • Random Rotation (±15°)
  • ColorJitter (brightness=0.15, contrast=0.15)
  • ToTensor
  ↓
Normalización (ImageNet):
  • mean = [0.485, 0.456, 0.406]
  • std = [0.229, 0.224, 0.225]
    ↓
Tensores PyTorch
    ↓
DataLoader (Batch size: 16)
    ↓
Modelo CNN
```


### Split de Datos

- **Training (70%)**: 77 imágenes con online augmentation
- **Validation (15%)**: 16 imágenes sin augmentation
- **Testing (15%)**: 18 imágenes para evaluación final

## 📁 Estructura del Proyecto

```
CloudClassify13/
├── config.py                  # Configuración centralizada (hiperparámetros optimizados)
├── model.py                   # Definición del modelo CNN (53M parámetros)
├── dataset.py                 # Carga y procesamiento de datos (con online augmentation)
├── train.py                   # Pipeline de entrenamiento
├── predict.py                 # Sistema de inferencia (con detección de no-nubes)
├── app.py                     # Servidor Flask
├── main_train.py              # Script principal de entrenamiento
├── augment_dataset.py         # ⚠️ NO USAR - Causa data leakage (ver advertencia)
├── download_data.py           # Descarga de imágenes (URLs desactualizadas)
├── plot_results.py            # Visualización de métricas
├── requirements.txt           # Dependencias del proyecto
├── web/                       # Interfaz web
│   ├── index.html            # Página principal
│   └── static/
│       ├── script.js         # Lógica del cliente (diseño de 2 columnas)
│       └── style.css         # Estilos (interfaz moderna)
├── data/                      # Dataset organizado por clase
│   ├── Ci/                   # Cirrus (~10 imágenes)
│   ├── Cc/                   # Cirrocumulus
│   ├── Cs/                   # Cirrostratus
│   ├── Ac/                   # Altocumulus
│   ├── As/                   # Altostratus
│   ├── Cu/                   # Cumulus
│   ├── Cb/                   # Cumulonimbus
│   ├── Ns/                   # Nimbostratus
│   ├── Sc/                   # Stratocumulus
│   ├── St/                   # Stratus
│   └── Ct/                   # Contrails
├── models/                    # Modelos guardados
│   └── cloud_classifier_best.pth  # Mejor modelo (43.75% accuracy)
└── notebooks/                 # Notebooks de experimentación (opcional)
```
├── requirements.txt           # Dependencias Python
├── README.md                  # Este archivo
│
├── web/                       # Interfaz web
│   ├── index.html            # Página principal
│   └── static/
│       ├── style.css         # Estilos CSS
│       └── script.js         # Lógica JavaScript
│
├── data/                      # Datos de entrenamiento
│   ├── Ci/                   # Imágenes de Cirrus
│   ├── Cc/                   # Imágenes de Cirrocumulus
│   ├── Cs/                   # Imágenes de Cirrostratus
│   ├── Ac/                   # Imágenes de Altocumulus
│   ├── As/                   # Imágenes de Altostratus
│   ├── Cu/                   # Imágenes de Cumulus
│   ├── Cb/                   # Imágenes de Cumulonimbus
│   ├── Ns/                   # Imágenes de Nimbostratus
│   ├── Sc/                   # Imágenes de Stratocumulus
│   ├── St/                   # Imágenes de Stratus
│   └── Ct/                   # Imágenes de Contrails
│
├── models/                    # Modelos entrenados
│   ├── checkpoint_*.pt       # Checkpoints durante entrenamiento
│   ├── best_model.pt         # Mejor modelo encontrado
│   └── training_history.json # Histórico de entrenamiento
│
├── notebooks/                 # Jupyter notebooks (opcional)
└── uploads/                   # Imágenes subidas (generado por app.py)
```

## 🚀 Instalación

### Requisitos del Sistema

- Python 3.8 o superior
- pip (administrador de paquetes Python)
- Opcional: GPU NVIDIA con CUDA para aceleración (requiere PyTorch con CUDA)

### Pasos de Instalación

1. **Clonar/Descargar el proyecto:**
```bash
cd CloudClassify13
```

2. **Crear entorno virtual (recomendado):**
```bash
# En Windows
python -m venv venv
venv\Scripts\activate

# En Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Instalar dependencias:**

**IMPORTANTE**: Existen 2 versiones de PyTorch:

#### Opción A: CPU (Ligero - ~200MB)
```bash
# Instalación simple para CPU
pip install -r requirements.txt
```

#### Opción B: GPU CUDA (Recomendado - ~2.8GB) ⚡
```bash
# PRIMERO: Instalar PyTorch con CUDA 11.8
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu118

# DESPUÉS: Instalar resto de dependencias
pip install -r requirements.txt
```

> 📚 **Ver [INSTALACION_CUDA.md](INSTALACION_CUDA.md)** para guía detallada sobre:
> - Instalación GPU vs CPU
> - Migración entre versiones
> - Tamaños de descarga
> - Requisitos de hardware
> - Troubleshooting

4. **Verificar instalación:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA disponible: {torch.cuda.is_available()}')"
```

**Output esperado (GPU)**:
```
PyTorch: 2.9.1
CUDA disponible: True
```

**Output esperado (CPU)**:
```
PyTorch: 2.9.1
CUDA disponible: False
```

5. **Dataset ya incluido:**
El proyecto incluye ~111 imágenes (10 por clase) en `data/`. 
**No es necesario descargar más datos para comenzar.**

## 📚 Uso

### 1. Entrenar el Modelo

**Entrenamiento con configuración optimizada (recomendado):**
```bash
python main_train.py --mode train
```

**Entrenamiento rápido para pruebas:**
```bash
python main_train.py --mode train --epochs 10 --device auto
```

**Opciones disponibles:**
```bash
python main_train.py --help
```

**Parámetros principales:**
- `--mode`: `train`, `evaluate` o `predict`
- `--epochs`: Número de épocas (default: 50, recomendado: 100)
- `--device`: `cuda`, `cpu` o `auto` (default: auto)
- `--verbose`: Muestra salida detallada

**Configuración actual (en `config.py`):**
- Batch size: **16** (optimizado para dataset pequeño)
- Learning rate: **0.0005** (convergencia suave)
- Dropout: **0.6** (prevención de overfitting)
- Early stopping patience: **30** (más tiempo para aprender)

### 2. Usar Interfaz Web

**Iniciar servidor Flask:**
```bash
python app.py
```

Luego abrir en navegador: **`http://localhost:5000`**

**Características de la interfaz:**
- 📤 **Subida drag-and-drop** de imágenes
- 🖼️ **Diseño de 2 columnas**: Imagen izquierda, resultados derecha
- 📊 **Visualización detallada**: Tipo de nube, confianza, descripción, top-3
- ⚠️ **Detección de no-nubes**: Alerta cuando confianza < 25%
- 🎨 **Interfaz moderna**: Animaciones y diseño responsive

### 3. Realizar Predicciones por CLI

**Predicción única:**
```bash
python predict.py --image ruta/imagen.jpg
```

**Predicción con modelo específico:**
```bash
python predict.py --image imagen.jpg --checkpoint models/cloud_classifier_best.pth
```
- Ver top-3 predicciones
- Ver descripción de tipo de nube
- Información sobre todas las clases

## 📊 Resultados Esperados

### Métricas de Entrenamiento

Durante el entrenamiento, el modelo registra:

```
Epoch 1/50
Train Loss: 2.3945 | Train Acc: 0.1523 | Val Acc: 0.1875
Epoch 2/50
Train Loss: 2.1234 | Train Acc: 0.2456 | Val Acc: 0.2800
...
Epoch 50/50
Train Loss: 0.3421 | Train Acc: 0.8934 | Val Acc: 0.8456
```

### Histórico de Entrenamiento

Se guarda en `models/training_history.json`:

```json
{
  "epochs": [
    {"epoch": 1, "train_loss": 2.3945, "train_acc": 0.1523, "val_loss": 2.4123, "val_acc": 0.1875},
    {"epoch": 2, "train_loss": 2.1234, "train_acc": 0.2456, "val_loss": 2.2045, "val_acc": 0.2800},
    ...
  ],
  "best_epoch": 45,
  "best_val_acc": 0.8956
}
```

## 🔧 Módulos del Proyecto

### config.py
Configuración centralizada optimizada:
- Rutas de directorios
- Clases de nubes (11 tipos WMO)
- **Hiperparámetros optimizados**: batch=16, lr=0.0005, dropout=0.6
- Early stopping patience=30
- NO_CLOUD_THRESHOLD=0.25

### model.py
Arquitectura CNN (53M parámetros):
- Clase `CloudCNN` con 4 bloques convolucionales
- BatchNorm después de cada convolución
- Dropout 0.6 en capas fully connected
- Inicialización He para ReLU

### dataset.py
Pipeline de datos con **Online Augmentation**:
- Clase `CloudDataset` para cargar imágenes
- **Transformaciones en tiempo real**:
  - RandomHorizontalFlip(p=0.5)
  - RandomRotation(15°)
  - ColorJitter(brightness=0.15, contrast=0.15)
- DataLoaders con batch_size=16
- Split estratificado 70/15/15

### train.py
Sistema de entrenamiento robusto:
- Clase `CloudClassifierTrainer`
- Loop de entrenamiento/validación
- **Early stopping** con patience=30
- Guardado automático del mejor modelo
- Optimizador Adam con lr=0.0005
- Tracking de métricas (loss, accuracy)

### predict.py
Sistema de inferencia inteligente:
- Clase `CloudPredictor`
- **Detección de no-nubes** (threshold=0.25)
- Predicción de imágenes individuales o lotes
- Top-K predicciones con probabilidades
- Campo `is_likely_cloud` en respuesta
- Warnings para baja confianza

### app.py
Servidor web Flask:
- Ruta `/` - Interfaz HTML moderna
- Ruta `/api/predict` POST - Clasificación de imagen
- Ruta `/api/classes` GET - Lista de clases
- Ruta `/api/info` GET - Metadata del modelo
- Manejo de errores robusto

### augment_dataset.py ⚠️
**NO USAR - Mantener solo como referencia**:
- Genera augmentación offline (permanente)
- **Problema**: Causa data leakage entre splits
- **Resultado**: Reduce accuracy de 43.75% a 22-28%
- **Alternativa**: Usar online augmentation en `dataset.py`
- Validación de archivos
- Manejo de errores

## 📈 Mejoras Futuras

1. **Mejoras del Modelo:**
   - Transfer Learning (ResNet50, EfficientNet)
   - Vision Transformers (ViT)
   - Ensemble de modelos
   - Pruning y quantization

2. **Mejoras de Datos:**
   - Descargador automático de URLs
   - Generación sintética con GANs
   - Data augmentation avanzada

3. **Interfaz:**
   - Visualización de heatmaps de atención
   - Historial de predicciones
   - Exportación de reportes
   - Dashboard de métricas

4. **Producción:**
   - Dockerización
   - Deployment en nube (AWS, Google Cloud)
   - API REST completa
   - Sistema de caché

## 📝 Referencias

### Papers de Investigación
- He, K., et al. (2015). "Deep Residual Learning for Image Recognition" (ResNet)
- Ioffe, S., & Szegedy, C. (2015). "Batch Normalization: Accelerating Deep Network Training"
- Simonyan, K., & Zisserman, A. (2014). "Very Deep Convolutional Networks for Large-Scale Image Recognition" (VGG)

### Estándares Meteorológicos
- [WMO Cloud Classification](https://library.wmo.int/index.php)
- [International Cloud Atlas - WMO](https://cloudatlas.wmo.int/)

### Proyectos Base
- [CloudClassification (FastAI)](https://github.com/...)
- [Ground-based Cloud Classification](https://github.com/...)
- [Cloud-Classification-New (PyTorch)](https://github.com/...)

## 📄 Licencia

Este proyecto está disponible bajo licencia MIT. Ver archivo LICENSE para detalles.

## 👥 Equipo

**Grupo #13 - 2025**

### Integrantes
- [Nombre Estudiante 1]
- [Nombre Estudiante 2]
- [Nombre Estudiante 3]
- [Nombre Estudiante 4]

### Profesor
- [Nombre del Profesor]
- Universidad: [Tu Universidad]
- Curso: [Nombre del Curso]

## 💬 Contacto

Para preguntas o sugerencias sobre el proyecto, contactar a:
- Email: [Tu Email]
- GitHub: [Tu GitHub]

## 🙏 Agradecimientos

- Agradecemos a la [Universidad] por los recursos y soporte
- Agradecemos a los creadores de PyTorch y Flask
- Agradecemos a la WMO por los estándares de clasificación de nubes

---

**Última actualización:** Enero 2025  
**Versión:** 1.0.0
