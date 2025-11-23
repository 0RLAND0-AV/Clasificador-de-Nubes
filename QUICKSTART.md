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

### 2. Descargar Imágenes de Ejemplo
```bash
python download_data.py --max-per-class 5
```

### 3. Entrenar Modelo (Rápido)
```bash
python main_train.py --mode train --epochs 5
```

### 4. Usar Interfaz Web
```bash
python app.py
```
Abrir: http://localhost:5000

---

## 📊 Entrenamiento Completo (30+ minutos)

```bash
# Con GPU (recomendado)
python main_train.py --mode train --epochs 50 --device cuda --verbose

# Con CPU (más lento)
python main_train.py --mode train --epochs 50 --device cpu
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
- Checkpoints guardados cada época

---

## 🐛 Troubleshooting

### "No se encuentran datos"
```bash
python download_data.py
# O agregar imágenes manualmente a data/
```

### "Out of memory" (GPU)
```bash
python main_train.py --mode train --batch-size 8 --device cuda
```

### "Module not found"
```bash
pip install --upgrade -r requirements.txt
```

### Puerto 5000 ya está en uso
```bash
# Cambiar puerto en app.py o usar:
python app.py --port 5001
```

---

## 📚 Comandos Completos

```bash
# Entrenar con todos los parámetros
python main_train.py --mode train \
  --epochs 100 \
  --batch-size 32 \
  --lr 0.001 \
  --device auto \
  --verbose

# Evaluar modelo
python main_train.py --mode evaluate \
  --checkpoint models/best_model.pt

# Predecir imagen
python main_train.py --mode predict \
  --image datos/nube.jpg \
  --checkpoint models/best_model.pt

# Descargar datos
python download_data.py \
  --data-dir data \
  --max-per-class 10 \
  --verbose

# Servidor web
python app.py
```

---

## 📖 Documentación

Ver `README.md` para documentación completa.

---

**¡A entrenar el modelo! 🚀**
