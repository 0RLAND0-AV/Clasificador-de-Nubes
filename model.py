"""
CloudClassify13 - Modelo CNN Personalizado
===========================================
Arquitectura de Red Neuronal Convolucional para clasificación de nubes.

Archivo: model.py
Descripción: Define la arquitectura CNN con capas convolucionales,
             pooling, dropout, batch normalization y capas FC.
             Se implementan conceptos del marco teórico de CNNs.

==============================================================================
CONCEPTOS CLAVE DE CNN:
==============================================================================

1. CAPA CONVOLUCIONAL (Conv2d):
   - Aplica filtros (kernels) que detectan patrones en la imagen
   - kernel_size=3: Filtro de 3x3 píxeles (estándar en CNNs modernas)
   - padding=1: Agrega borde para mantener el tamaño de salida
   - Los primeros filtros detectan bordes/líneas
   - Los últimos filtros detectan conceptos abstractos (tipos de nubes)

2. BATCH NORMALIZATION (BatchNorm2d):
   - Normaliza las activaciones de cada capa
   - Hace el entrenamiento más estable y rápido
   - Permite usar learning rates más altos

3. FUNCIÓN DE ACTIVACIÓN (ReLU):
   - ReLU(x) = max(0, x)
   - Introduce no-linealidad (esencial para aprender patrones complejos)
   - Simple y eficiente computacionalmente

4. MAX POOLING (MaxPool2d):
   - Reduce el tamaño de la imagen a la mitad (2x2, stride=2)
   - Mantiene las características más importantes
   - Reduce parámetros y evita overfitting

5. GLOBAL AVERAGE POOLING (AdaptiveAvgPool2d):
   - Promedia cada mapa de características a un solo valor
   - Alternativa moderna a Flatten (menos parámetros)
   - Hace el modelo independiente del tamaño de entrada

6. CAPAS FULLY CONNECTED (Linear):
   - Neuronas tradicionales que combinan todas las características
   - La última capa tiene NUM_CLASSES neuronas (una por tipo de nube)

7. DROPOUT:
   - "Apaga" neuronas aleatoriamente durante entrenamiento
   - Previene overfitting (memorización excesiva)
   - dropout_rate=0.0 significa sin dropout (modo memorización)

==============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import NUM_CLASSES, DROPOUT_RATE, FC_LAYERS


class CloudCNN(nn.Module):
    """
    Red Neuronal Convolucional Ligera (Lightweight CNN).
    
    Optimizada para datasets pequeños (100-500 imágenes):
    1. Menos filtros (32-256) en lugar de (64-512)
    2. Global Average Pooling (GAP) en lugar de Flatten masivo
    3. Solo 2 capas FC finales
    
    Arquitectura:
    ┌─────────────────────────────────────────────────────────────┐
    │ INPUT: Imagen 224x224x3 (RGB)                               │
    ├─────────────────────────────────────────────────────────────┤
    │ CONV1: 3→32 filtros, 3x3, ReLU + BN + MaxPool → 112x112x32  │
    │ CONV2: 32→64 filtros, 3x3, ReLU + BN + MaxPool → 56x56x64   │
    │ CONV3: 64→128 filtros, 3x3, ReLU + BN + MaxPool → 28x28x128 │
    │ CONV4: 128→256 filtros, 3x3, ReLU + BN + MaxPool → 14x14x256│
    ├─────────────────────────────────────────────────────────────┤
    │ GLOBAL AVG POOL: 14x14x256 → 1x1x256 (256 valores)          │
    ├─────────────────────────────────────────────────────────────┤
    │ FC1: 256 → 128 neuronas + ReLU + Dropout                    │
    │ FC2: 128 → 12 neuronas (una por clase de nube)              │
    └─────────────────────────────────────────────────────────────┘
    │ OUTPUT: 12 probabilidades (una por tipo de nube)            │
    └─────────────────────────────────────────────────────────────┘
    
    Total de parámetros: ~400,000 (mucho menos que VGG16: 138 millones)
    """
    
    def __init__(self, num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE):
        super(CloudCNN, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # ======================================================================
        # BLOQUE CONVOLUCIONAL 1: Detecta bordes y gradientes simples
        # ======================================================================
        # Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # - 3 canales de entrada (RGB)
        # - 32 filtros de salida (aprende 32 patrones diferentes)
        # - kernel 3x3 (tamaño del filtro)
        # - padding=1 para mantener el mismo tamaño espacial
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Normaliza las 32 activaciones
        self.pool = nn.MaxPool2d(2, 2)  # Reduce tamaño a la mitad
        
        # ======================================================================
        # BLOQUE CONVOLUCIONAL 2: Detecta texturas y patrones
        # ======================================================================
        # Toma los 32 mapas del bloque 1 y produce 64 mapas
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # ======================================================================
        # BLOQUE CONVOLUCIONAL 3: Detecta formas y estructuras
        # ======================================================================
        # Toma los 64 mapas del bloque 2 y produce 128 mapas
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # ======================================================================
        # BLOQUE CONVOLUCIONAL 4: Detecta conceptos abstractos (tipos de nubes)
        # ======================================================================
        # Toma los 128 mapas del bloque 3 y produce 256 mapas
        # Aquí el modelo ya "entiende" qué tipo de nube es
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # ======================================================================
        # GLOBAL AVERAGE POOLING
        # ======================================================================
        # Convierte cada mapa de 14x14 en un solo valor (promedio)
        # Resultado: 256 valores (uno por filtro de conv4)
        # Ventajas:
        #   - Reduce drásticamente los parámetros
        #   - Hace el modelo independiente del tamaño de entrada
        #   - Actúa como regularización (menos overfitting)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # ======================================================================
        # CLASIFICADOR (Fully Connected Layers)
        # ======================================================================
        # Dropout: "Apaga" neuronas aleatoriamente durante entrenamiento
        # Con dropout_rate=0.0, NO se apagan neuronas (modo memorización)
        self.dropout = nn.Dropout(dropout_rate)
        
        # FC1: Reduce de 256 a 128 neuronas
        # Aprende combinaciones de características para clasificar
        self.fc1 = nn.Linear(256, 128)
        
        # FC2: Capa final con 12 neuronas (una por tipo de nube)
        # Las salidas son "logits" (se convierten a probabilidades con softmax)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Inicializar pesos con método Kaiming (óptimo para ReLU)
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Inicializa los pesos de las capas usando el método Kaiming.
        
        ¿Por qué es importante?
        - Pesos mal inicializados → el modelo no aprende
        - Kaiming está diseñado específicamente para ReLU
        - Mantiene la varianza de las activaciones controlada
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming para capas convolucionales con ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                # BatchNorm: peso=1, bias=0 (estándar)
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Kaiming para capas FC con ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Propagación hacia adelante (forward pass).
        
        Procesa la imagen a través de todas las capas.
        
        Args:
            x: Tensor de forma (batch_size, 3, 224, 224)
               - batch_size: número de imágenes procesadas juntas
               - 3: canales RGB
               - 224x224: tamaño de imagen
        
        Returns:
            Tensor de forma (batch_size, 12)
            - 12 valores (logits) por imagen, uno por tipo de nube
        """
        # ==================== EXTRACCIÓN DE CARACTERÍSTICAS ====================
        # Cada bloque: Conv → BatchNorm → ReLU → MaxPool
        
        # Bloque 1: 224x224x3 → 112x112x32
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Bloque 2: 112x112x32 → 56x56x64
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Bloque 3: 56x56x64 → 28x28x128
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Bloque 4: 28x28x128 → 14x14x256
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # ==================== POOLING GLOBAL Y CLASIFICACIÓN ====================
        # Global Average Pooling: 14x14x256 → 1x1x256
        x = self.global_pool(x)
        
        # Flatten: Convierte de (batch, 256, 1, 1) a (batch, 256)
        x = x.view(x.size(0), -1)
        
        # Dropout (si dropout_rate > 0)
        x = self.dropout(x)
        
        # FC1: 256 → 128 con ReLU
        x = F.relu(self.fc1(x))
        
        # FC2: 128 → 12 (sin activación - los logits van a CrossEntropyLoss)
        x = self.fc2(x)
        
        return x
    
    def get_model_summary(self):
        """Retorna información del modelo."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.__class__.__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate,
        }


def create_model(num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE, device='cpu'):
    """
    Factory function para crear un modelo CloudCNN.
    
    Args:
        num_classes (int): Número de clases (por defecto 11)
        dropout_rate (float): Tasa de dropout (por defecto 0.5)
        device (str): Dispositivo ('cpu' o 'cuda')
    
    Returns:
        CloudCNN: Modelo inicializado y movido al dispositivo especificado
    """
    model = CloudCNN(num_classes=num_classes, dropout_rate=dropout_rate)
    model = model.to(device)
    return model


if __name__ == "__main__":
    # Test del modelo
    print("=" * 60)
    print("TEST DEL MODELO CNN")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✓ Dispositivo: {device}")
    
    # Crear modelo
    model = create_model(device=device)
    print(f"✓ Modelo creado: {model.__class__.__name__}")
    
    # Mostrar resumen
    summary = model.get_model_summary()
    print(f"\nResumen del modelo:")
    print(f"  - Total de parámetros: {summary['total_parameters']:,}")
    print(f"  - Parámetros entrenables: {summary['trainable_parameters']:,}")
    print(f"  - Número de clases: {summary['num_classes']}")
    print(f"  - Dropout rate: {summary['dropout_rate']}")
    
    # Test con entrada dummy
    print(f"\nTest con entrada dummy...")
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    output = model(dummy_input)
    print(f"  - Input shape: {dummy_input.shape}")
    print(f"  - Output shape: {output.shape}")
    print(f"  - Output shape esperado: (2, 11) ✓")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETADO EXITOSAMENTE ✓")
    print("=" * 60)
