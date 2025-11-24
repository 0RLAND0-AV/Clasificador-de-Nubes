# CloudClassify13 - Guía Rápida

## ⚡ Inicio Rápido (5 minutos)

### 1. Instalación Básica
```bash
# Crear entorno virtual
python -m venv venv
venv\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Agregar Datos (OPCIONAL) NO EJECUTES ESTE PASO TODAVIA, LAS URLS NO SIRVEN.
```bash
# OPCIÓN A: Descargar imágenes de ejemplo (URLs pueden estar desactualizadas)
python download_data.py --max-per-class 5

# OPCIÓN B (RECOMENDADO): Agregar tus propias imágenes manualmente
# Copiar imágenes a las carpetas: data/Ci/, data/Cc/, data/Cs/, etc.
# Mínimo 10 imágenes por clase, formato JPG/PNG
```

> **⚠️ NOTA:** El proyecto ya incluye 10 imágenes por clase en la carpeta `data/`.
> Este paso es opcional si deseas agregar más imágenes de entrenamiento.

### 3. Entrenar Modelo

#### Entrenamiento Básico (modo por defecto)
```bash
# Entrena con configuración por defecto (50 épocas)
python main_train.py
```

#### Entrenamiento Rápido (para pruebas) **USA ESTE ES MAS RAPIDO**
```bash
# Solo 5 épocas para prueba rápida
python main_train.py --mode train --epochs 5 --verbose
```

#### Entrenamiento Completo (recomendado)
```bash
# Entrenamiento completo con GPU y salida detallada
python main_train.py --mode train --epochs 50 --device auto --verbose
```

#### Parámetros Disponibles:
- `--mode`: Modo de operación
  - `train` (entrenar modelo)
  - `evaluate` (evaluar modelo existente)
  - `predict` (predecir una imagen)
- `--epochs`: Número de épocas de entrenamiento (default: 50)
- `--batch-size`: Tamaño del batch (default: 32)
- `--lr`: Tasa de aprendizaje (default: 0.001)
- `--device`: Dispositivo de cómputo
  - `auto` (GPU si disponible, sino CPU)
  - `cuda` (forzar GPU)
  - `cpu` (forzar CPU)
- `--verbose`: Mostrar información detallada del entrenamiento
- `--checkpoint`: Ruta a checkpoint existente (para evaluar/predecir)

### 4. Usar Interfaz Web
```bash
python app.py
```
Abrir: http://localhost:5000

---

## 📊 Ejemplos de Uso Completos

### Entrenar con GPU y 100 épocas
```bash
python main_train.py --mode train --epochs 100 --batch-size 32 --lr 0.001 --device cuda --verbose
```

### Entrenar con CPU (más lento)
```bash
python main_train.py --mode train --epochs 50 --device cpu --verbose
```

### Evaluar modelo guardado
```bash
python main_train.py --mode evaluate --checkpoint models/cloud_classifier_best.pth
```

### Predecir una imagen
```bash
python main_train.py --mode predict --image ruta/mi_nube.jpg
```

### Predecir con checkpoint específico
```bash
python main_train.py --mode predict --image ruta/mi_nube.jpg --checkpoint models/cloud_classifier_best.pth
```

---

## 🔮 Predicciones

### Imagen Individual
```bash
python main_train.py --mode predict --image ruta/imagen.jpg
```

### Interfaz Web
```bash
python app.py
# Luego: Drag-and-drop imagen en http://localhost:5000
```

---

## 🗂️ Agregar más Imágenes

Estructura esperada:
```
CloudClassify13/
└── data/
    ├── Ci/      (Cirrus)
    ├── Cc/      (Cirrocumulus)
    ├── Cs/      (Cirrostratus)
    ├── Ac/      (Altocumulus)
    ├── As/      (Altostratus)
    ├── Cu/      (Cumulus)
    ├── Cb/      (Cumulonimbus)
    ├── Ns/      (Nimbostratus)
    ├── Sc/      (Stratocumulus)
    ├── St/      (Stratus)
    └── Ct/      (Contrails)
```

Copiar imágenes JPG/PNG en las carpetas correspondientes.

---

## 💡 Tips

### Rendimiento
- **GPU recomendada**: NVIDIA GPU con CUDA 11.8+
- **Requisito mínimo**: 4GB RAM, 1GB almacenamiento
- **Imágenes ideales**: 224×224px, JPG, PNG

### Datos
- Mínimo 10 imágenes por clase para entrenar
- Máximo: 100+ imágenes por clase para mejor accuracy
- Distribución balanceada mejora resultados

### Entrenamiento
- Early stopping detiene entrenamiento si no mejora
- Augmentation automática previene overfitting
- Checkpoints guardados cada época (el mejor se guarda automáticamente)

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
