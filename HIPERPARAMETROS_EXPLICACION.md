# 📘 Explicación de Hiperparámetros - CloudClassify13

## 🎯 ¿Qué son los Hiperparámetros?

Los hiperparámetros son **configuraciones que tú decides ANTES** de entrenar el modelo. A diferencia de los parámetros del modelo (pesos y sesgos que se aprenden automáticamente), **tú controlas los hiperparámetros** para afectar cómo aprende la red.

---

## 🔧 Hiperparámetros Principales

### 1. BATCH_SIZE (Tamaño del Lote)

**¿Qué es?**  
Número de imágenes que el modelo procesa **simultáneamente** antes de actualizar sus pesos.

```python
BATCH_SIZE = 16  # Procesa 16 imágenes a la vez
```

#### 📊 Analogía
Imagina que estudias para un examen:
- **Batch pequeño (8-16)**: Lees 1 página y haces resumen inmediatamente
- **Batch grande (32-64)**: Lees 10 páginas y luego haces resumen de todo

#### Efectos de Diferentes Valores

| Valor | Efecto | Ventajas | Desventajas | Uso Recomendado |
|-------|--------|----------|-------------|-----------------|
| **4-8** | Muy pequeño | ✅ Menos memoria GPU<br>✅ Actualizaciones frecuentes<br>✅ Escapa mínimos locales | ❌ Entrenamiento inestable<br>❌ Muy ruidoso<br>❌ Más lento | GPU pequeña (2GB VRAM) |
| **16** ⭐ | Pequeño | ✅ Buen balance<br>✅ Funciona con datasets pequeños<br>✅ Regularización implícita | ❌ Un poco ruidoso | **Dataset pequeño (111 imgs)** ← TU CASO |
| **32** | Medio | ✅ Entrenamiento estable<br>✅ Buen compromiso | ❌ Necesita más memoria<br>❌ Puede sobreajustar con pocos datos | Dataset mediano (500-1000 imgs) |
| **64-128** | Grande | ✅ Muy estable<br>✅ Rápido por época | ❌ Mucha memoria GPU<br>❌ Puede quedar atascado<br>❌ Sobreajuste | Dataset grande (5000+ imgs) |
| **1** | Extremo | ⚠️ Actualiza con cada imagen | ❌ Muy inestable<br>❌ No aprovecha paralelismo | ❌ NO usar |
| **>128** | Extremo | ⚠️ Procesa muchas imágenes | ❌ Memoria insuficiente<br>❌ Gradientes muy suaves | ❌ NO usar en PC normal |

#### 🧪 Particiones de Equivalencia para BATCH_SIZE

```
┌─────────────────────────────────────────────────────────────────┐
│ PARTICIONES DE EQUIVALENCIA - BATCH_SIZE                        │
├─────────────────────────────────────────────────────────────────┤
│ Inválidos      │ < 1                 │ ❌ ERROR: No tiene sentido│
├─────────────────────────────────────────────────────────────────┤
│ Muy Pequeño    │ 1-7                 │ ⚠️ Demasiado ruidoso      │
├─────────────────────────────────────────────────────────────────┤
│ Pequeño ✅     │ 8-16                │ ✅ ÓPTIMO para datos<111  │
├─────────────────────────────────────────────────────────────────┤
│ Medio          │ 17-48               │ ✅ BUENO para datos<1000  │
├─────────────────────────────────────────────────────────────────┤
│ Grande         │ 49-128              │ ✅ BUENO para datos>5000  │
├─────────────────────────────────────────────────────────────────┤
│ Muy Grande ⚠️  │ >128                │ ⚠️ CUDA Out of Memory     │
└─────────────────────────────────────────────────────────────────┘
```

#### Por qué cambiaste de 32 a 16:

**ANTES (32)**:
```
Dataset: 111 imágenes
Train: 77 imágenes → 77/32 = 2.4 batches por época
                   → Solo 2 actualizaciones de pesos
```
❌ Muy pocas actualizaciones para aprender bien

**AHORA (16)**:
```
Dataset: 111 imágenes  
Train: 77 imágenes → 77/16 = 4.8 batches por época
                   → 4-5 actualizaciones de pesos
```
✅ Más actualizaciones = mejor aprendizaje

---

### 2. LEARNING_RATE (Tasa de Aprendizaje)

**¿Qué es?**  
Qué tan grande es el "paso" que da el modelo al actualizar sus pesos.

```python
LEARNING_RATE = 0.0005  # Pasos pequeños y cuidadosos
```

#### 📊 Analogía
Imaginas que estás bajando una montaña con los ojos vendados:
- **LR alto (0.01)**: Das pasos GRANDES → Rápido pero peligroso (puedes caerte)
- **LR bajo (0.0001)**: Das pasos PEQUEÑOS → Seguro pero lento
- **LR óptimo (0.0005)**: Pasos medianos → Equilibrio perfecto

#### Efectos Visuales

```
LEARNING RATE MUY ALTO (0.1):
Loss
  │     ╱╲     ╱╲     ╱╲
  │    ╱  ╲   ╱  ╲   ╱  ╲
  │   ╱    ╲ ╱    ╲ ╱    ╲
  └──────────────────────────► Época
  ❌ Oscila y nunca converge (salta demasiado)


LEARNING RATE ÓPTIMO (0.0005):
Loss
  │╲
  │ ╲___
  │     ╲___
  │         ╲___
  └──────────────────────────► Época
  ✅ Desciende suavemente al mínimo


LEARNING RATE MUY BAJO (0.00001):
Loss
  │╲
  │ ╲
  │  ╲
  │   ╲
  └──────────────────────────► Época
  ⚠️ Muy lento, tarda 1000 épocas
```

#### 🧪 Particiones de Equivalencia para LEARNING_RATE

```
┌─────────────────────────────────────────────────────────────────┐
│ PARTICIONES DE EQUIVALENCIA - LEARNING_RATE                     │
├─────────────────────────────────────────────────────────────────┤
│ Inválido       │ < 0                 │ ❌ ERROR: Negativo sube   │
├─────────────────────────────────────────────────────────────────┤
│ Muy Bajo       │ 0.00001-0.0001      │ ⚠️ Demasiado lento        │
├─────────────────────────────────────────────────────────────────┤
│ Bajo ✅        │ 0.0001-0.0005       │ ✅ ÓPTIMO: Estable        │
├─────────────────────────────────────────────────────────────────┤
│ Medio          │ 0.0005-0.002        │ ✅ BUENO: Rápido          │
├─────────────────────────────────────────────────────────────────┤
│ Alto ⚠️        │ 0.002-0.01          │ ⚠️ Puede oscilar          │
├─────────────────────────────────────────────────────────────────┤
│ Muy Alto ❌    │ > 0.01              │ ❌ Diverge, no converge   │
└─────────────────────────────────────────────────────────────────┘
```

| Valor | Comportamiento | Resultado |
|-------|----------------|-----------|
| **0.00001** | Pasos minúsculos | ⏱️ Tarda 1000 épocas en aprender |
| **0.0001** | Pasos pequeños | ✅ Estable pero lento |
| **0.0005** ⭐ | Pasos medianos | ✅ **ÓPTIMO: Equilibrio perfecto** |
| **0.001** | Pasos normales | ✅ Funciona bien en la mayoría |
| **0.01** | Pasos grandes | ⚠️ Oscila, puede no converger |
| **0.1** | Pasos enormes | ❌ Diverge completamente |
| **Negativo** | Sube en vez de bajar | ❌ ERROR: El modelo empeora |

#### Por qué cambiaste de 0.001 a 0.0005:

**ANTES (0.001)**:
```
Época 1: Loss = 2.5
Época 2: Loss = 1.8  (bajó 0.7)
Época 3: Loss = 1.9  (¡subió!) ← Oscila
Época 4: Loss = 1.7
```
⚠️ Oscilaba porque daba pasos muy grandes

**AHORA (0.0005)**:
```
Época 1: Loss = 2.5
Época 2: Loss = 2.1  (bajó 0.4)
Época 3: Loss = 1.8  (bajó 0.3)
Época 4: Loss = 1.6  (bajó 0.2) ← Descenso suave
```
✅ Descenso suave y constante

---

### 3. DROPOUT_RATE (Tasa de Dropout)

**¿Qué es?**  
Porcentaje de neuronas que se "apagan" aleatoriamente durante el entrenamiento para evitar overfitting.

```python
DROPOUT_RATE = 0.6  # Apaga el 60% de neuronas aleatoriamente
```

#### 📊 Analogía
Imagina un equipo de fútbol practicando:
- **Dropout 0.0**: Todos juegan siempre → Se acostumbran mucho entre ellos (overfitting)
- **Dropout 0.6**: Solo 4 de 10 jugadores por entrenamiento → Aprenden a adaptarse
- **Dropout 0.9**: Solo 1 jugador entrena → No pueden aprender nada

#### Efectos Visuales

```
DROPOUT = 0.0 (Sin dropout):
Train Accuracy: 99% ✅
Val Accuracy:   40% ❌  ← OVERFITTING
└─ Memorizó el dataset pero no generaliza


DROPOUT = 0.6 (Óptimo):
Train Accuracy: 75% ✅
Val Accuracy:   44% ✅  ← GENERALIZA BIEN
└─ Aprendió patrones generales


DROPOUT = 0.9 (Demasiado alto):
Train Accuracy: 30% ❌
Val Accuracy:   28% ❌  ← UNDERFITTING
└─ No pudo aprender nada
```

#### 🧪 Particiones de Equivalencia para DROPOUT_RATE

```
┌─────────────────────────────────────────────────────────────────┐
│ PARTICIONES DE EQUIVALENCIA - DROPOUT_RATE                      │
├─────────────────────────────────────────────────────────────────┤
│ Inválido       │ < 0 o > 1           │ ❌ ERROR: Debe ser 0-1    │
├─────────────────────────────────────────────────────────────────┤
│ Sin Dropout    │ 0.0-0.2             │ ⚠️ Overfitting probable   │
├─────────────────────────────────────────────────────────────────┤
│ Bajo           │ 0.2-0.4             │ ✅ Dataset grande (>5000) │
├─────────────────────────────────────────────────────────────────┤
│ Medio          │ 0.4-0.6             │ ✅ Dataset mediano        │
├─────────────────────────────────────────────────────────────────┤
│ Alto ✅        │ 0.6-0.7             │ ✅ **Dataset pequeño<500**│
├─────────────────────────────────────────────────────────────────┤
│ Muy Alto ⚠️    │ 0.7-0.9             │ ⚠️ Underfitting posible   │
├─────────────────────────────────────────────────────────────────┤
│ Extremo ❌     │ 0.9-1.0             │ ❌ Modelo no aprende      │
└─────────────────────────────────────────────────────────────────┘
```

| Valor | Neuronas Activas | Resultado |
|-------|------------------|-----------|
| **0.0** | 100% | ❌ Overfitting: Memoriza datos |
| **0.3** | 70% | ✅ Dataset grande (5000+ imgs) |
| **0.5** | 50% | ✅ Dataset mediano (1000 imgs) |
| **0.6** ⭐ | 40% | ✅ **Dataset pequeño (111 imgs)** ← TU CASO |
| **0.8** | 20% | ⚠️ Demasiada regularización |
| **0.95** | 5% | ❌ Modelo no puede aprender |
| **Negativo** | N/A | ❌ ERROR |

---

### 4. EPOCHS (Épocas)

**¿Qué es?**  
Número de veces que el modelo ve **TODOS** los datos de entrenamiento.

```python
EPOCHS = 100  # El modelo verá las 77 imágenes 100 veces
```

#### 📊 Analogía
Estudiar para un examen:
- **1 época**: Lees el libro una vez
- **10 épocas**: Lees el libro 10 veces
- **100 épocas**: Lees el libro 100 veces

#### Efectos

```
POCAS ÉPOCAS (10):
Accuracy
   │         ╱
   │       ╱
   │     ╱
   │   ╱
   └───────────► Época
   ⚠️ Modelo no terminó de aprender


ÉPOCAS ÓPTIMAS (50-100):
Accuracy
   │       ┌────
   │     ╱
   │   ╱
   │ ╱
   └───────────► Época
   ✅ Modelo aprendió y se estabilizó


DEMASIADAS ÉPOCAS (500):
Accuracy
Train │           ╱────
Val   │       ╱──┐
      │     ╱    │ ↓ Empeora
      │   ╱      ↓
      └───────────► Época
   ❌ Overfitting: Memorizó datos
```

#### 🧪 Particiones de Equivalencia para EPOCHS

```
┌─────────────────────────────────────────────────────────────────┐
│ PARTICIONES DE EQUIVALENCIA - EPOCHS                            │
├─────────────────────────────────────────────────────────────────┤
│ Inválido       │ < 1                 │ ❌ ERROR: No entrena      │
├─────────────────────────────────────────────────────────────────┤
│ Muy Poco       │ 1-10                │ ⚠️ No aprende suficiente  │
├─────────────────────────────────────────────────────────────────┤
│ Poco           │ 11-30               │ ⚠️ Puede no converger     │
├─────────────────────────────────────────────────────────────────┤
│ Óptimo ✅      │ 50-150              │ ✅ BUENO: Con Early Stop  │
├─────────────────────────────────────────────────────────────────┤
│ Muchas         │ 150-300             │ ⚠️ Ineficiente            │
├─────────────────────────────────────────────────────────────────┤
│ Excesivas ❌   │ > 300               │ ❌ Overfitting garantizado│
└─────────────────────────────────────────────────────────────────┘
```

**NOTA**: Con **Early Stopping** (patience=30), el entrenamiento se detiene automáticamente cuando no mejora, así que puedes poner 100 épocas sin riesgo.

---

### 5. EARLY_STOPPING_PATIENCE (Paciencia)

**¿Qué es?**  
Cuántas épocas esperar sin mejora antes de detener el entrenamiento automáticamente.

```python
EARLY_STOPPING_PATIENCE = 30  # Espera 30 épocas sin mejora
```

#### 📊 Analogía
Esperando a un amigo que llega tarde:
- **Patience = 5**: Esperas 5 minutos y te vas
- **Patience = 30**: Esperas 30 minutos (más paciencia)
- **Patience = 100**: Esperas eternamente (inútil)

#### Efectos

```
PATIENCE = 5 (Muy Bajo):
Época 10: Val Acc = 40% (mejor)
Época 11: Val Acc = 38%
Época 12: Val Acc = 39%
Época 13: Val Acc = 38%
Época 14: Val Acc = 39%
Época 15: Val Acc = 38%
└─ STOP ❌ (Se detuvo muy pronto, podría haber mejorado)


PATIENCE = 30 (Óptimo):
Época 10: Val Acc = 40% (mejor)
Época 11-39: Val Acc = 37-39% (oscilando)
Época 40: Val Acc = 44% (¡mejoró!) ✅
└─ Continuó y encontró mejor modelo
```

#### 🧪 Particiones de Equivalencia para PATIENCE

```
┌─────────────────────────────────────────────────────────────────┐
│ PARTICIONES DE EQUIVALENCIA - EARLY_STOPPING_PATIENCE          │
├─────────────────────────────────────────────────────────────────┤
│ Inválido       │ < 1                 │ ❌ ERROR: Detiene al toque│
├─────────────────────────────────────────────────────────────────┤
│ Muy Bajo       │ 1-5                 │ ⚠️ Detiene muy pronto     │
├─────────────────────────────────────────────────────────────────┤
│ Bajo           │ 6-15                │ ⚠️ Puede perder mejoras   │
├─────────────────────────────────────────────────────────────────┤
│ Medio ✅       │ 16-30               │ ✅ ÓPTIMO: Balance        │
├─────────────────────────────────────────────────────────────────┤
│ Alto           │ 31-50               │ ⚠️ Desperdicia tiempo     │
├─────────────────────────────────────────────────────────────────┤
│ Muy Alto ❌    │ > 50                │ ❌ Prácticamente sin stop │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧪 Tabla Resumen - Particiones de Equivalencia

| Hiperparámetro | Inválido | Muy Bajo | Bajo/Óptimo | Medio | Alto | Muy Alto |
|----------------|----------|----------|-------------|-------|------|----------|
| **BATCH_SIZE** | <1 ❌ | 1-7 ⚠️ | **8-16 ✅** | 17-48 | 49-128 | >128 ❌ |
| **LEARNING_RATE** | <0 ❌ | 0.00001 ⚠️ | **0.0005 ✅** | 0.001-0.002 | 0.01 ⚠️ | >0.01 ❌ |
| **DROPOUT_RATE** | <0 o >1 ❌ | 0.0-0.2 ⚠️ | 0.3-0.5 | **0.6 ✅** | 0.7-0.8 ⚠️ | 0.9-1.0 ❌ |
| **EPOCHS** | <1 ❌ | 1-10 ⚠️ | 11-30 ⚠️ | **50-150 ✅** | 150-300 ⚠️ | >300 ❌ |
| **PATIENCE** | <1 ❌ | 1-5 ⚠️ | 6-15 ⚠️ | **16-30 ✅** | 31-50 ⚠️ | >50 ❌ |

✅ = Valores óptimos para tu caso (111 imágenes)

---

## 💡 Configuración Actual Explicada

```python
# ==================== TU CONFIGURACIÓN ACTUAL ====================

BATCH_SIZE = 16
# ✅ ÓPTIMO para 111 imágenes
# → 77 train / 16 = ~5 batches por época
# → 5 actualizaciones de pesos por época
# Antes era 32 → Solo 2 actualizaciones (muy poco)

LEARNING_RATE = 0.0005
# ✅ ÓPTIMO: Pasos pequeños y estables
# → Descenso suave sin oscilaciones
# Antes era 0.001 → Oscilaba demasiado

DROPOUT_RATE = 0.6
# ✅ ÓPTIMO para dataset pequeño
# → Apaga 60% de neuronas aleatoriamente
# → Previene overfitting (memorización)
# Antes era 0.5 → No era suficiente

EPOCHS = 100
# ✅ SUFICIENTE con early stopping
# → Permite entrenar completamente
# → Early stopping lo detiene si no mejora

EARLY_STOPPING_PATIENCE = 30
# ✅ ÓPTIMO: Da tiempo suficiente
# → Espera 30 épocas sin mejora
# → No detiene prematuramente
# Antes era 25 → A veces detenía muy pronto
```

---

## 🎯 ¿Qué pasa si cambias los valores?

### Escenario 1: BATCH_SIZE = 4 (Muy pequeño)
```
✅ Ventajas:
- Funciona en GPU pequeña (2GB)
- 77/4 = 19 actualizaciones por época (muchas)

❌ Desventajas:
- Entrenamiento muy ruidoso e inestable
- Loss oscila mucho
- Tarda más tiempo
- Puede no converger
```

### Escenario 2: BATCH_SIZE = 64 (Muy grande)
```
❌ Problemas:
- CUDA out of memory (GPU insuficiente)
- 77/64 = 1.2 batches por época (muy poco)
- Solo 1 actualización por época
- No aprende nada
- Overfitting garantizado
```

### Escenario 3: LEARNING_RATE = 0.01 (Muy alto)
```
❌ Resultado:
Época 1: Loss = 2.5
Época 2: Loss = 3.8  ← Subió en vez de bajar
Época 3: Loss = 1.2
Época 4: Loss = 4.1  ← Oscila violentamente
Época 5: Loss = 2.7
└─ DIVERGE: Nunca converge
```

### Escenario 4: LEARNING_RATE = 0.00001 (Muy bajo)
```
⏱️ Resultado:
Época 1: Loss = 2.5000
Época 2: Loss = 2.4995  ← Baja muy poco
Época 3: Loss = 2.4990
Época 4: Loss = 2.4985
...
Época 100: Loss = 2.4500 ← Todavía no terminó
└─ LENTO: Tardaría 1000 épocas
```

### Escenario 5: DROPOUT = 0.0 (Sin regularización)
```
❌ Resultado:
Train Accuracy: 99% ✅ (Aparentemente perfecto)
Val Accuracy:   25% ❌ (Peor que azar)

└─ OVERFITTING: Memorizó las 77 imágenes
   pero no puede clasificar nuevas imágenes
```

### Escenario 6: DROPOUT = 0.9 (Demasiada regularización)
```
❌ Resultado:
Train Accuracy: 20%
Val Accuracy:   18%

└─ UNDERFITTING: El 90% de neuronas apagadas
   no deja aprender al modelo
```

---

## 📊 Valores Negativos/Extremos

### BATCH_SIZE Negativo o 0
```python
BATCH_SIZE = -16  # ❌ ERROR
BATCH_SIZE = 0    # ❌ ERROR

# Python lanza excepción:
ValueError: batch_size should be a positive integer value, but got batch_size=-16
```

### LEARNING_RATE Negativo
```python
LEARNING_RATE = -0.001  # ❌ ERROR CONCEPTUAL

# Qué pasa:
# El gradiente se invierte
# El modelo SUBE en vez de BAJAR
# Loss aumenta en vez de disminuir
# 
# Época 1: Loss = 2.5
# Época 2: Loss = 5.8  ← PEOR
# Época 3: Loss = 12.3 ← PEOR
# Época 4: Loss = 45.7 ← PEOR
```

### DROPOUT Fuera de Rango
```python
DROPOUT_RATE = 1.5   # ❌ ERROR
DROPOUT_RATE = -0.3  # ❌ ERROR

# Python lanza excepción:
ValueError: dropout probability has to be between 0 and 1, but got 1.5
```

### EPOCHS = 0 o Negativo
```python
EPOCHS = 0   # ❌ No entrena nada
EPOCHS = -10 # ❌ ERROR

# El modelo no se entrena
# Accuracy permanece aleatoria (~9% para 11 clases)
```

---

## 🎓 Recomendaciones Finales

### Para TU proyecto (111 imágenes):
```python
# ✅ CONFIGURACIÓN ÓPTIMA ACTUAL
BATCH_SIZE = 16              # Balance perfecto
LEARNING_RATE = 0.0005       # Estable
DROPOUT_RATE = 0.6           # Previene overfitting
EPOCHS = 100                 # Con early stopping
EARLY_STOPPING_PATIENCE = 30 # Tiempo suficiente
```

### Si tuvieras MÁS datos (1000 imágenes):
```python
BATCH_SIZE = 32              # Más estable
LEARNING_RATE = 0.001        # Más rápido
DROPOUT_RATE = 0.5           # Menos regularización
EPOCHS = 150
EARLY_STOPPING_PATIENCE = 20
```

### Si tuvieras MUCHOS datos (10,000 imágenes):
```python
BATCH_SIZE = 64
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.3
EPOCHS = 200
EARLY_STOPPING_PATIENCE = 15
```

---

## 📚 Resumen Ejecutivo

1. **BATCH_SIZE**: Cuántas imágenes procesa a la vez
   - Pequeño (16) → Más actualizaciones → Mejor para pocos datos ✅

2. **LEARNING_RATE**: Tamaño del paso al aprender
   - Bajo (0.0005) → Aprendizaje lento pero estable ✅

3. **DROPOUT**: Neuronas apagadas aleatoriamente
   - Alto (0.6) → Previene memorización con pocos datos ✅

4. **EPOCHS**: Veces que ve todos los datos
   - 100 es suficiente con early stopping ✅

5. **PATIENCE**: Épocas a esperar sin mejora
   - 30 da tiempo suficiente para mejorar ✅

**Tu configuración actual es ÓPTIMA para 111 imágenes** ⭐
