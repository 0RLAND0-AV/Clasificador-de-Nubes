"""
CloudClassify13 - Entrenamiento del Modelo
===========================================
Loop de entrenamiento, validación y gestión de checkpoints.

Archivo: train.py
Descripción: Implementa el ciclo completo de entrenamiento:
             - Forward pass
             - Cálculo de pérdida
             - Backpropagation
             - Early stopping
             - Guardado de checkpoints
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR
import time
from pathlib import Path
import json

from config import (
    EPOCHS, LEARNING_RATE, WEIGHT_DECAY, OPTIMIZER_TYPE, LOSS_FUNCTION,
    EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_DELTA, USE_LR_SCHEDULER,
    LR_SCHEDULER_TYPE, LR_SCHEDULER_T_MAX, LOG_FREQUENCY, MODEL_PATH,
    MODELS_DIR
)


class CloudClassifierTrainer:
    """
    Clase para entrenar el modelo de clasificación de nubes.
    
    Características:
    - Entrenamiento con validación
    - Early stopping automático
    - Learning rate scheduling
    - Guardado de checkpoints
    - Logging de métricas
    """
    
    def __init__(self, model, device='cpu', learning_rate=LEARNING_RATE, 
                 optimizer_name=None, scheduler_name=None, checkpoint_dir=None, verbose=False):
        """
        Inicializa el entrenador.
        
        Args:
            model: Modelo a entrenar
            device: Dispositivo ('cpu' o 'cuda')
            learning_rate: Tasa de aprendizaje inicial
            optimizer_name: Nombre del optimizer ('adam', 'sgd', 'rmsprop')
            scheduler_name: Nombre del scheduler ('cosine', 'step', 'exponential')
            checkpoint_dir: Directorio para guardar checkpoints
            verbose: Modo verbose
        """
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.checkpoint_dir = checkpoint_dir or MODELS_DIR
        self.best_checkpoint_path = None
        
        # Determinar optimizer
        optimizer_type = optimizer_name or OPTIMIZER_TYPE
        
        # Configurar optimizer
        if optimizer_type.lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=WEIGHT_DECAY
            )
        elif OPTIMIZER_TYPE.lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=WEIGHT_DECAY
            )
        elif OPTIMIZER_TYPE.lower() == 'rmsprop':
            self.optimizer = optim.RMSprop(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=WEIGHT_DECAY
            )
        else:
            raise ValueError(f"Optimizer no soportado: {optimizer_type}")
        
        # Determinar scheduler
        scheduler_type = scheduler_name or LR_SCHEDULER_TYPE
        
        # Configurar scheduler (opcional)
        self.scheduler = None
        if USE_LR_SCHEDULER:
            if scheduler_type == 'cosine':
                self.scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=LR_SCHEDULER_T_MAX
                )
            elif scheduler_type == 'step':
                self.scheduler = StepLR(
                    self.optimizer,
                    step_size=10,
                    gamma=0.1
                )
            elif scheduler_type == 'exponential':
                self.scheduler = ExponentialLR(
                    self.optimizer,
                    gamma=0.95
                )
        
        # Criterio de pérdida
        if LOSS_FUNCTION.lower() == 'crossentropy':
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Loss function no soportada: {LOSS_FUNCTION}")
        
        # Historial de entrenamiento
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Early stopping
        self.early_stopping_counter = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
    
    def train_epoch(self, train_loader):
        """
        Entrena una época.
        
        Args:
            train_loader: DataLoader de entrenamiento
        
        Returns:
            tuple: (pérdida promedio, precisión promedio)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Estadísticas
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Logging
            if (batch_idx + 1) % LOG_FREQUENCY == 0:
                current_loss = total_loss / (batch_idx + 1)
                current_acc = 100. * correct / total
                print(f"  Batch {batch_idx+1:4d} | Loss: {current_loss:.4f} | Acc: {current_acc:.2f}%")
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = 100. * correct / total
        
        return avg_loss, avg_acc
    
    def validate(self, val_loader):
        """
        Valida el modelo.
        
        Args:
            val_loader: DataLoader de validación
        
        Returns:
            tuple: (pérdida promedio, precisión promedio)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        avg_acc = 100. * correct / total
        
        return avg_loss, avg_acc
    
    def train(self, train_loader, val_loader, epochs=EPOCHS):
        """
        Entrena el modelo durante múltiples épocas.
        
        Args:
            train_loader: DataLoader de entrenamiento
            val_loader: DataLoader de validación
            epochs: Número de épocas
        """
        print("\n" + "=" * 70)
        print("INICIANDO ENTRENAMIENTO DEL MODELO")
        print("=" * 70)
        print(f"Epochs: {epochs} | Learning Rate: {self.learning_rate}")
        print(f"Optimizer: {OPTIMIZER_TYPE} | Loss: {LOSS_FUNCTION}")
        print(f"Early Stopping: {EARLY_STOPPING_PATIENCE} | LR Scheduler: {USE_LR_SCHEDULER}")
        print("=" * 70 + "\n")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            print(f"Epoch [{epoch+1}/{epochs}]")
            
            # Entrenar
            train_loss, train_acc = self.train_epoch(train_loader)
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            
            # Validar
            val_loss, val_acc = self.validate(val_loader)
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Scheduler step
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                print(f"  LR: {current_lr:.6f}")
            
            # Guardar historial
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Guardar mejor modelo
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.early_stopping_counter = 0
                self._save_checkpoint(epoch, best=True)
                print(f"  ✓ Nuevo mejor modelo guardado (Acc: {val_acc:.2f}%)")
            else:
                self.early_stopping_counter += 1
                print(f"  Early stopping: {self.early_stopping_counter}/{EARLY_STOPPING_PATIENCE}")
                
                if self.early_stopping_counter >= EARLY_STOPPING_PATIENCE:
                    print(f"\n⚠ Early stopping activado en época {epoch+1}")
                    break
            
            print()
        
        # Resumen final
        elapsed_time = time.time() - start_time
        print("=" * 70)
        print("ENTRENAMIENTO COMPLETADO")
        print(f"Tiempo total: {elapsed_time:.1f}s ({elapsed_time/60:.1f}m)")
        print(f"Mejor validación Acc: {self.best_val_acc:.2f}%")
        print(f"Mejor validación Loss: {self.best_val_loss:.4f}")
        print("=" * 70 + "\n")
        
        return self.history
    
    def _save_checkpoint(self, epoch, best=False):
        """
        Guarda un checkpoint del modelo.
        
        Args:
            epoch: Número de época
            best: Si es True, guarda como mejor modelo
        """
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        if best:
            checkpoint_path = Path(self.checkpoint_dir) / "cloud_classifier_best.pth"
            self.best_checkpoint_path = str(checkpoint_path)
        else:
            checkpoint_path = Path(self.checkpoint_dir) / f"checkpoint_epoch_{epoch+1}.pth"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }, checkpoint_path)
    
    def save_model(self, path=None):
        """Guarda el modelo entrenado."""
        if path is None:
            path = Path(self.checkpoint_dir) / "cloud_classifier_best.pth"
        
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"✓ Modelo guardado en: {path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Carga un checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"✓ Checkpoint cargado desde: {checkpoint_path}")
    
    def save_history(self, path=None):
        """Guarda el historial de entrenamiento."""
        if path is None:
            path = Path(self.checkpoint_dir) / "training_history.json"
        
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"✓ Historial guardado en: {path}")


if __name__ == "__main__":
    print("Módulo de entrenamiento importado correctamente ✓")
