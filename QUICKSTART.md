# CloudClassify13 - Guía Rápida


### 1. Instalación Básica
```bash
# Crear y activar entorno virtual
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

**IMPORTANTE - Elegir versión de PyTorch**:

#### Opción A: CPU (Ligero - ~200MB) 💻
```bash
# Instalación simple - funciona en cualquier PC
pip install -r requirements.txt
```

#### Opción B: GPU CUDA 11.8 (Recomendado - ~2.8GB) ⚡
```bash
# PRIMERO: PyTorch con CUDA (~2.8GB - puede tardar 10-30 min)
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu118

# DESPUÉS: Resto de dependencias
pip install -r requirements.txt
```
> 💡 **¿No sabes cuál elegir?**  
> - Si tienes GPU NVIDIA → Opción B (vale la pena la descarga)
> - Si no tienes GPU o solo vas a probar → Opción A
> 
> 📚 **Más info**: Ver [INSTALACION_CUDA.md](INSTALACION_CUDA.md)

### 3. Entrenar Modelo

#### ⭐ Opción A: Entrenamiento (Recomendado para empezar)
```bash
# Entrenamiento completo con GPU (200 épocas)
python main_train.py 

```

#### 🚀 Opción B: Entrenamiento rapido
```bash
# Entrenamiento de prueba rápido (10 épocas)
python main_train.py --mode train --epochs 10 --device auto --verbose
```


### 4. Usar Interfaz Web
```bash
python app.py
```
Abrir en navegador: **http://localhost:5000**

**Características**:
- 🖼️ Diseño de 2 columnas (imagen izq, resultados der)
- 📤 Drag & drop para subir imágenes
- 📊 Visualización con confianza y top-3 predicciones
- ⚠️ Alertas para imágenes sin nubes (confianza < 25%)

---

## 📊 Ejemplos de Uso Completos

### Caso 1: Entrenamiento Completo con GPU
```bash
# Mejor configuración para training completo
python main_train.py --mode train --epochs 200 --device cuda --verbose
```
### Caso 2: Entrenamiento Rápido (CPU)
```bash
# Para probar sin GPU (más lento)
python main_train.py --mode train --epochs 10 --device cpu --verbose
```

### Caso 4: Predicción CLI Individual
```bash
# Predecir una imagen específica
python predict.py --image ruta/mi_nube.jpg
```

**Output esperado**:
```json
{
  "predicted_class": "Cu",
  "confidence": 0.567,
  "is_likely_cloud": true,
  "top_predictions": [
    {"class": "Cu", "probability": 0.567},
    {"class": "Sc", "probability": 0.234},
    {"class": "Ac", "probability": 0.123}
  ]
}
```


## 🗂️ Agregar Más Imágenes (Opcional)

Si quieres **mejorar el accuracy**, necesitas más datos reales:

### Estructura de Carpetas
```
CloudClassify13/
└── data/
    ├── Ci/      # Cirrus (~10 imágenes incluidas)
    ├── Cc/      # Cirrocumulus
    ├── Cs/      # Cirrostratus
    ├── Ac/      # Altocumulus
    ├── As/      # Altostratus
    ├── Cu/      # Cumulus
    ├── Cb/      # Cumulonimbus
    ├── Ns/      # Nimbostratus
    ├── Sc/      # Stratocumulus
    ├── St/      # Stratus
    └── Ct/      # Contrails
```

### Cómo Agregar Imágenes
1. **Manual** (Recomendado):
   - Descargar imágenes de Google Images, Flickr, etc.
   - Renombrar: `Cu_012.jpg`, `Ci_045.png`
   - Copiar a carpeta correspondiente

2. **Datasets Públicos**:
   - SWIM-CCSN Dataset
   - MGCD (Multimodal Ground-based Cloud Dataset)
   - CloudSeg Dataset

**Objetivo**: 50-100+ imágenes por clase para accuracy > 70%

---

## 💡 Tips y Mejores Prácticas

### 🎯 Rendimiento
- **GPU recomendada**: NVIDIA con CUDA 11.8+ (10x más rápido)
- **CPU aceptable**: Funciona pero más lento (30-60 min por época)
- **Requisitos mínimos**: 4GB RAM, 2GB disco libre
- **Imágenes**: JPG/PNG, 224×224px (se redimensionan automáticamente)

### 📊 Dataset
- **Actual**: 111 imágenes → **43.75% accuracy** (limitado)
- **Recomendado**: 500-1000 imágenes → 70-85% accuracy esperado
- **Óptimo**: 5000+ imágenes → 90%+ accuracy posible
- **Balance**: Misma cantidad de imágenes por clase

### 🚀 Entrenamiento
- **Early stopping** se activa automáticamente (patience=30)
- **Online augmentation** funciona en tiempo real
- **Mejor modelo** se guarda automáticamente
- **Checkpoints**: Guardan progreso cada época

### 🐛 Troubleshooting Común

**Problema**: `RuntimeError: CUDA out of memory`
```bash
# Solución: Reducir batch size en config.py
BATCH_SIZE = 8  # o 4 para GPUs pequeñas
```

**Problema**: Entrenamiento muy lento en CPU
```bash
# Solución: Reducir épocas o usar GPU
python main_train.py --mode train --epochs 10 --device cpu
```

**Problema**: Accuracy no mejora de ~43%
```bash
# Causa: Dataset muy pequeño (111 imágenes)
# Solución: Agregar más imágenes reales (500+ por clase)
```

**Problema**: Error "No module named 'torch'"
```bash
# Solución: Reinstalar PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 🐛 Troubleshooting

### "No se encuentran datos"
```bash
# Verificar que existen imágenes en data/
ls data/Ci/
# Si está vacío, agregar imágenes manualmente o ejecutar:
python download_data.py
```

### "Out of memory" (GPU)
```bash
# Reducir batch size
python main_train.py --mode train --batch-size 8 --device cuda
```

### "Module not found: tensorboard"
```bash
# Instalar dependencias faltantes
pip install tensorboard matplotlib
```

### "Module not found"
```bash
# Reinstalar todas las dependencias
pip install --upgrade -r requirements.txt
```

### Puerto 5000 ya está en uso
```bash
# Editar app.py y cambiar WEB_PORT en config.py
# O matar el proceso: netstat -ano | findstr :5000
```

---

## 📚 Documentación

Ver `README.md` para documentación completa.

---

**¡A entrenar el modelo! 🚀**
