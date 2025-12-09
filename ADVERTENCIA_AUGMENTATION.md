# ⚠️ ADVERTENCIA: augment_dataset.py

## 🚫 NO USAR ESTE SCRIPT

El archivo `augment_dataset.py` está incluido en el proyecto **SOLO COMO REFERENCIA EDUCATIVA**.

**NO debe ejecutarse** porque causa problemas graves de data leakage y reduce significativamente el accuracy del modelo.

---

## ❌ Problema: Data Leakage

### ¿Qué es Data Leakage?

El **data leakage** ocurre cuando información del conjunto de validación o test "se filtra" al conjunto de entrenamiento, causando que el modelo:
- Aparente mejor rendimiento del real
- Falle al generalizar a datos nuevos
- Memorice variaciones en lugar de aprender patrones

### ¿Cómo causa leakage `augment_dataset.py`?

```
┌─────────────────────────────────────────────────────────────┐
│ PASO 1: Augmentación Offline (augment_dataset.py)          │
│                                                             │
│ Imagen Original:                                            │
│   data/Cu/Cu_001.jpg                                        │
│                                                             │
│ Genera 10 copias aumentadas:                                │
│   data/Cu/Cu_001_aug_0.jpg  (flip horizontal)              │
│   data/Cu/Cu_001_aug_1.jpg  (rotación 10°)                 │
│   data/Cu/Cu_001_aug_2.jpg  (brillo +15%)                  │
│   ...                                                       │
│   data/Cu/Cu_001_aug_9.jpg  (combinación)                  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ PASO 2: Split de datos (dataset.py)                        │
│                                                             │
│ Las 11 imágenes (1 original + 10 aumentadas) se            │
│ distribuyen ALEATORIAMENTE:                                 │
│                                                             │
│   TRAIN:      Cu_001.jpg, Cu_001_aug_2.jpg, Cu_001_aug_5   │
│   VALIDATION: Cu_001_aug_1.jpg, Cu_001_aug_8.jpg           │
│   TEST:       Cu_001_aug_3.jpg                             │
│                                                             │
│ ❌ PROBLEMA: El modelo ve la "misma nube" en train,        │
│             validation y test con pequeñas variaciones      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ RESULTADO: Data Leakage                                     │
│                                                             │
│ • El modelo "memoriza" las nubes específicas               │
│ • Accuracy en validación es artificialmente alta           │
│ • Pero falla con imágenes realmente nuevas                 │
│ • Accuracy real baja drásticamente                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Resultados Experimentales

Probamos `augment_dataset.py` con 10x augmentation en el dataset:

| Experimento | Método | Imágenes | Val Accuracy | Resultado |
|-------------|--------|----------|--------------|-----------|
| **1** | Sin augmentation | 111 | **37.5%** | Baseline |
| **2** | Offline aug (10x) | 1,110 | **22.95%** | ❌ Peor |
| **3** | Offline aug (5x) | 555 | **28.28%** | ❌ Peor |
| **4** | Online aug | 111 | **43.75%** | ✅ Mejor |

**Conclusión**: Offline augmentation reduce accuracy en **15-20 puntos porcentuales** por data leakage.

---

## ✅ Alternativa Correcta: Online Augmentation

La **online augmentation** (implementada en `dataset.py`) resuelve el problema:

```python
# dataset.py (YA IMPLEMENTADO)
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),      # 50% probabilidad
    transforms.RandomRotation(15),               # ±15 grados
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

### Ventajas de Online Augmentation

| Aspecto | Online (✅) | Offline (❌) |
|---------|------------|--------------|
| **Timing** | Durante entrenamiento | Antes de entrenar |
| **Almacenamiento** | No usa espacio | Multiplica archivos |
| **Data Leakage** | ❌ No ocurre | ✅ Sí ocurre |
| **Variedad** | Infinita (random cada época) | Finita (archivos fijos) |
| **Accuracy** | 43.75% | 22-28% |
| **Split** | Limpio | Contaminado |

### Cómo Funciona Online Augmentation

```
┌────────────────────────────────────────────────────┐
│ ENTRENAMIENTO CON ONLINE AUGMENTATION             │
│                                                    │
│ Split ANTES de augmentation:                      │
│   TRAIN:      Cu_001.jpg, Cu_003.jpg, Cu_005.jpg  │
│   VALIDATION: Cu_002.jpg, Cu_006.jpg              │
│   TEST:       Cu_004.jpg                          │
│                                                    │
│ Durante cada época:                                │
│   Época 1: Cu_001.jpg → flip + rotate 5°          │
│   Época 2: Cu_001.jpg → NO flip + rotate -12°     │
│   Época 3: Cu_001.jpg → flip + rotate 3° + brillo │
│   ...                                              │
│                                                    │
│ ✅ Validation y Test NUNCA se modifican           │
│ ✅ Train ve variaciones diferentes cada época     │
│ ✅ No hay leakage entre splits                    │
└────────────────────────────────────────────────────┘
```

---

## 🎓 Lección Aprendida

### ¿Por qué existe `augment_dataset.py` en el proyecto?

1. **Histórico**: Se creó inicialmente para aumentar el dataset pequeño
2. **Experimental**: Se probó como alternativa rápida
3. **Educativo**: Se mantiene para mostrar el problema de data leakage
4. **Advertencia**: Ejemplo de qué NO hacer en ML

### ¿Cuándo podría ser útil offline augmentation?

**SOLO en casos muy específicos**:
- Dataset ya está dividido manualmente en train/val/test
- Augmentation se aplica SOLO a train, nunca a val/test
- Se necesita pre-procesar una vez por eficiencia computacional
- Se tiene control total del pipeline

**En nuestro caso**: NO aplica porque `dataset.py` hace split automático.

---

## 📝 Recomendaciones

### Si necesitas más datos:

1. **Opción A (Mejor)**: Agregar imágenes reales
   ```bash
   # Buscar datasets públicos:
   # - SWIM-CCSN Cloud Dataset
   # - MGCD (Multimodal Ground-based Cloud Dataset)
   # - CloudSeg Dataset
   ```

2. **Opción B (Ya implementada)**: Usar online augmentation
   ```python
   # Ya está en dataset.py, no requiere cambios
   ```

3. **Opción C (NO recomendada)**: Transfer Learning
   ```python
   # Usar modelo pre-entrenado (ResNet, VGG, EfficientNet)
   # Requiere modificar model.py
   ```

### Si quieres experimentar con augmentation:

```python
# Modificar dataset.py (líneas 30-40)
# Aumentar intensidad de transformaciones:

transforms.RandomHorizontalFlip(p=0.7),      # Más agresivo
transforms.RandomRotation(25),               # Mayor rotación
transforms.ColorJitter(
    brightness=0.25,                         # Mayor variación
    contrast=0.25,
    saturation=0.15,
    hue=0.05
),
transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Agregar
```

---

## 🔍 Cómo Detectar Data Leakage

### Síntomas:
- ✅ Accuracy de entrenamiento: 95%
- ❌ Accuracy de validación: 90%
- ❌ Accuracy con imágenes nuevas: 25%

### Diagnóstico:
```python
# Verificar si hay imágenes similares entre splits
import os
from PIL import Image
import imagehash

def check_leakage(data_dir):
    hashes = {}
    for split in ['train', 'val', 'test']:
        for img_path in get_images(split):
            img = Image.open(img_path)
            h = imagehash.average_hash(img)
            if h in hashes:
                print(f"⚠️ LEAKAGE: {img_path} similar a {hashes[h]}")
            hashes[h] = img_path
```

---

## 📚 Referencias

- [Data Leakage in Machine Learning](https://machinelearningmastery.com/data-leakage-machine-learning/)
- [Data Augmentation Best Practices](https://www.tensorflow.org/tutorials/images/data_augmentation)
- [Common ML Mistakes - Data Leakage](https://towardsdatascience.com/data-leakage-in-machine-learning-how-it-can-be-detected-and-minimize-the-risk-8ef4e3a97562)

---

## ✅ Resumen

| ¿Usar `augment_dataset.py`? | ❌ **NO** |
|-----------------------------|-----------|
| **Razón** | Causa data leakage |
| **Impacto** | Reduce accuracy 15-20% |
| **Alternativa** | Online augmentation (ya implementado) |
| **Ubicación** | `dataset.py` líneas 30-50 |
| **Estado del script** | Mantener solo como referencia |

**Mensaje final**: Si ves `augment_dataset.py`, **no lo ejecutes**. El proyecto ya tiene la solución correcta implementada.
