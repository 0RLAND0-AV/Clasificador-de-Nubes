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
    Red Neuronal Convolucional Ligera (Lightweight CNN).
    
    Optimizada para datasets pequeños (100-500 imágenes):
    1. Menos filtros (16-128) en lugar de (64-512)
    2. Global Average Pooling (GAP) en lugar de Flatten masivo
    3. Solo 1 capa FC final
    
    Reducción de parámetros: ~50,000,000 -> ~20,000
    """
    
    def __init__(self, num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE):
        super(CloudCNN, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Bloque 1: 16 filtros
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Bloque 2: 32 filtros
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Bloque 3: 64 filtros
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Bloque 4: 128 filtros
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Clasificador final (con una capa oculta extra para mejor capacidad)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Feature Extraction
        x = self.pool(F.relu(self.bn1(self.conv1(x)))) # 224 -> 112
        x = self.pool(F.relu(self.bn2(self.conv2(x)))) # 112 -> 56
        x = self.pool(F.relu(self.bn3(self.conv3(x)))) # 56 -> 28
        x = self.pool(F.relu(self.bn4(self.conv4(x)))) # 28 -> 14
        
        # Global Pooling & Classification
        x = self.global_pool(x)         # 128x14x14 -> 128x1x1
        x = x.view(x.size(0), -1)       # Flatten: 128
        
        x = self.dropout(x)
        x = F.relu(self.fc1(x))         # 128 -> 64
        x = self.fc2(x)                 # 64 -> 11
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
