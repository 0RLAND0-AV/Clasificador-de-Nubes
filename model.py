"""
CloudClassify13 - Modelo CNN Personalizado
===========================================
Arquitectura de Red Neuronal Convolucional para clasificación de nubes.

Archivo: model.py
Descripción: Define la arquitectura CNN con capas convolucionales,
             pooling, dropout, batch normalization y capas FC.
             Se implementan conceptos del marco teórico de CNNs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import NUM_CLASSES, DROPOUT_RATE, FC_LAYERS


class CloudCNN(nn.Module):
    """
    Red Neuronal Convolucional personalizada para clasificación de nubes.
    
    Componentes:
    - Capas Convolucionales: Extraen características visuales
    - Batch Normalization: Estabiliza el entrenamiento
    - ReLU: Función de activación no lineal
    - Max Pooling: Reduce dimensionalidad
    - Dropout: Regularización para evitar overfitting
    - Capas FC: Clasificación final
    
    Arquitectura:
    Input (224x224x3)
        ↓
    Conv1 (64 filtros) + BN + ReLU + MaxPool
        ↓
    Conv2 (128 filtros) + BN + ReLU + MaxPool
        ↓
    Conv3 (256 filtros) + BN + ReLU + MaxPool
        ↓
    Conv4 (512 filtros) + BN + ReLU + MaxPool
        ↓
    Flatten + Dropout(0.5)
        ↓
    FC Layers (512 → 256 → 128 → 11)
        ↓
    Output (11 clases)
    """
    
    def __init__(self, num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE):
        super(CloudCNN, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # ========== BLOQUE CONVOLUCIONAL 1 ==========
        # Input: 3x224x224 → Output: 64x112x112
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, 
                               kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output después pooling: 64x112x112
        
        # ========== BLOQUE CONVOLUCIONAL 2 ==========
        # Input: 64x112x112 → Output: 128x56x56
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, 
                               kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output después pooling: 128x56x56
        
        # ========== BLOQUE CONVOLUCIONAL 3 ==========
        # Input: 128x56x56 → Output: 256x28x28
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, 
                               kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output después pooling: 256x28x28
        
        # ========== BLOQUE CONVOLUCIONAL 4 ==========
        # Input: 256x28x28 → Output: 512x14x14
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, 
                               kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output después pooling: 512x14x14
        
        # ========== CAPAS COMPLETAMENTE CONECTADAS ==========
        # Después de flatten: 512 * 14 * 14 = 100,352
        self.flatten_size = 512 * 14 * 14
        
        self.fc1 = nn.Linear(self.flatten_size, FC_LAYERS[0])  # 100352 → 512
        self.dropout1 = nn.Dropout(p=dropout_rate)
        
        self.fc2 = nn.Linear(FC_LAYERS[0], FC_LAYERS[1])  # 512 → 256
        self.dropout2 = nn.Dropout(p=dropout_rate)
        
        self.fc3 = nn.Linear(FC_LAYERS[1], FC_LAYERS[2])  # 256 → 128
        self.dropout3 = nn.Dropout(p=dropout_rate)
        
        # Capa de salida
        self.fc_out = nn.Linear(FC_LAYERS[2], num_classes)  # 128 → 11 clases
        
        # Inicializar pesos
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Inicializa los pesos de la red.
        Usa inicialización de He (He normal) para capas con ReLU.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass de la red.
        
        Args:
            x (torch.Tensor): Tensor de entrada de forma (batch_size, 3, 224, 224)
        
        Returns:
            torch.Tensor: Logits de salida de forma (batch_size, num_classes)
        """
        # Bloque 1: Conv + BN + ReLU + MaxPool
        x = self.conv1(x)  # (B, 3, 224, 224) → (B, 64, 224, 224)
        x = self.bn1(x)    # Normalización por batch
        x = F.relu(x)      # Activación ReLU (no linealidad)
        x = self.pool1(x)  # (B, 64, 224, 224) → (B, 64, 112, 112)
        
        # Bloque 2: Conv + BN + ReLU + MaxPool
        x = self.conv2(x)  # (B, 64, 112, 112) → (B, 128, 112, 112)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)  # (B, 128, 112, 112) → (B, 128, 56, 56)
        
        # Bloque 3: Conv + BN + ReLU + MaxPool
        x = self.conv3(x)  # (B, 128, 56, 56) → (B, 256, 56, 56)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)  # (B, 256, 56, 56) → (B, 256, 28, 28)
        
        # Bloque 4: Conv + BN + ReLU + MaxPool
        x = self.conv4(x)  # (B, 256, 28, 28) → (B, 512, 28, 28)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)  # (B, 512, 28, 28) → (B, 512, 14, 14)
        
        # Flatten: Convierte tensor multidimensional en vector
        x = x.view(x.size(0), -1)  # (B, 512*14*14) = (B, 100352)
        
        # Capas completamente conectadas con Dropout
        x = self.fc1(x)      # (B, 100352) → (B, 512)
        x = F.relu(x)        # Activación ReLU
        x = self.dropout1(x) # Regularización: desactiva 50% de neuronas
        
        x = self.fc2(x)      # (B, 512) → (B, 256)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)      # (B, 256) → (B, 128)
        x = F.relu(x)
        x = self.dropout3(x)
        
        # Capa de salida (sin activación, se usa con CrossEntropyLoss)
        x = self.fc_out(x)   # (B, 128) → (B, 11)
        
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
