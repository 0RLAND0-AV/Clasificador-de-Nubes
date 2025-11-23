# CloudClassify13 - Clasificador de Nubes con CNN

**Proyecto de Grupo #13**  
Universidad: [Tu Universidad]  
Curso: Inteligencia Artificial / Machine Learning  
Año: 2025

## 📋 Descripción

CloudClassify13 es un sistema de clasificación automática de tipos de nubes basado en redes neuronales convolucionales (CNN). El proyecto combina un backend de machine learning en PyTorch con una interfaz web HTML/CSS/JavaScript para clasificar imágenes de nubes en 11 categorías estándar de la Organización Meteorológica Mundial (OMM/WMO).

### Características Principales

- ✅ **CNN Custom**: Red neuronal convolucional diseñada específicamente para clasificación de nubes
- ✅ **11 Clases de Nubes**: Clasificación según estándares WMO/OMM
- ✅ **Interfaz Web**: Carga de imágenes y visualización de resultados en tiempo real
- ✅ **API REST**: Endpoints para integración en otras aplicaciones
- ✅ **Pipeline Modular**: Código organizado en módulos independientes
- ✅ **Data Augmentation**: Técnicas de aumentación de datos para mejor generalización
- ✅ **Early Stopping**: Prevención de overfitting durante entrenamiento
- ✅ **GPU/CPU**: Soporte automático para aceleración GPU (CUDA)

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
├── FC1: Linear(100352, 512) → ReLU → Dropout(0.5)
├── FC2: Linear(512, 256) → ReLU → Dropout(0.5)
├── FC3: Linear(256, 128) → ReLU → Dropout(0.5)
└── Output: Linear(128, 11) → Logits (sin Softmax, se aplica en CrossEntropyLoss)
```

**Parámetros:**
- Total: ~100,000 parámetros
- Entrada: 224×224 RGB
- Salida: 11 clases

### Pipeline de Datos

```
Raw Images (224×224 RGB)
    ↓
Transformaciones (Train):
  • Resize a 224×224
  • Random Horizontal Flip
  • Random Rotation (±15°)
  • Random Crop
  • ColorJitter (brightness, contrast, saturation, hue)
  ↓
Normalización (ImageNet):
  • mean = [0.485, 0.456, 0.406]
  • std = [0.229, 0.224, 0.225]
    ↓
Tensores PyTorch
    ↓
DataLoader (Batch size: 32)
    ↓
Modelo CNN
```

### Split de Datos

- **Training (70%)**: Datos de entrenamiento con augmentation
- **Validation (15%)**: Datos de validación sin augmentation
- **Testing (15%)**: Evaluación final

## 📁 Estructura del Proyecto

```
CloudClassify13/
├── config.py                  # Configuración centralizada
├── model.py                   # Definición del modelo CNN
├── dataset.py                 # Carga y procesamiento de datos
├── train.py                   # Pipeline de entrenamiento
├── predict.py                 # Sistema de inferencia
├── app.py                     # Servidor Flask
├── main_train.py              # Script principal
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

- Python 3.7 o superior
- pip (administrador de paquetes Python)
- Opcional: GPU NVIDIA para aceleración CUDA

### Pasos de Instalación

1. **Clonar/Descargar el proyecto:**
```bash
cd tu/ruta/CloudClassify13
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
```bash
pip install -r requirements.txt
```

4. **Descargar o preparar imágenes de entrenamiento:**
```bash
# Las imágenes deben organizarse en estructura:
# data/Ci/*.jpg
# data/Cc/*.jpg
# ... etc
```

## 📚 Uso

### 1. Entrenar el Modelo

**Entrenamiento básico:**
```bash
python main_train.py --mode train
```

**Con parámetros personalizados:**
```bash
python main_train.py --mode train --epochs 100 --batch-size 16 --lr 0.0005
```

**Opciones disponibles:**
```bash
python main_train.py --mode train --help
```

**Parámetros:**
- `--epochs`: Número de épocas (default: 50)
- `--batch-size`: Tamaño de batch (default: 32)
- `--lr`: Tasa de aprendizaje (default: 0.001)
- `--device`: Dispositivo 'cuda', 'cpu' o 'auto' (default: auto)
- `--verbose`: Salida detallada

**Ejemplo con GPU:**
```bash
python main_train.py --mode train --epochs 100 --device cuda --verbose
```

### 2. Evaluar Modelo

```bash
python main_train.py --mode evaluate --checkpoint models/best_model.pt
```

### 3. Realizar Predicciones

**Predicción en imagen única:**
```bash
python main_train.py --mode predict --image ruta/a/imagen.jpg
```

**Con checkpoint específico:**
```bash
python main_train.py --mode predict --image imagen.jpg --checkpoint models/best_model.pt
```

### 4. Usar Interfaz Web

**Iniciar servidor Flask:**
```bash
python app.py
```

Luego abrir en navegador: `http://localhost:5000`

**Características:**
- Subir imagen via drag-and-drop
- Ver predicción en tiempo real
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
Configuración centralizada del proyecto:
- Rutas de directorios
- Clases de nubes
- Hiperparámetros del modelo
- Parámetros de entrenamiento
- Configuración de augmentation

### model.py
Define la arquitectura CNN:
- Clase `CloudCNN` con 4 bloques convolucionales
- Batch Normalization
- Dropout para regularización
- Inicialización He

### dataset.py
Pipeline de datos:
- Clase `CloudDataset` para cargar imágenes
- Transformaciones y augmentation
- DataLoaders para train/val/test
- Split estratificado

### train.py
Sistema de entrenamiento:
- Clase `CloudClassifierTrainer`
- Loop de entrenamiento y validación
- Early stopping
- Guardado de checkpoints
- Optimizadores configurables (Adam, SGD, RMSprop)
- Schedulers de learning rate

### predict.py
Sistema de inferencia:
- Clase `CloudPredictor`
- Predicción de imágenes individuales
- Predicción por lotes
- Top-K predicciones
- Generación de probabilidades

### app.py
Servidor web Flask:
- Ruta `/` para interfaz HTML
- Ruta `/api/predict` POST para clasificación
- Ruta `/api/classes` GET para listar clases
- Ruta `/api/info` GET para metadata
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
