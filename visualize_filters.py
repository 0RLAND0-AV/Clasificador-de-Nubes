"""
CloudClassify13 - Visualización de Filtros CNN
===============================================
Muestra qué ve cada capa convolucional al procesar una imagen.

Archivo: visualize_filters.py
Descripción: Genera visualizaciones de los feature maps (activaciones)
             de cada capa convolucional para entender qué aprende el modelo.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
import os

from config import MODEL_PATH, CLASS_NAMES, CLOUD_CLASSES, DATA_DIR, IMAGE_SIZE
from dataset import get_single_image_loader
from model import CloudCNN


class FilterVisualizer:
    """Visualiza los feature maps de cada capa convolucional."""
    
    def __init__(self, model_path=MODEL_PATH):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CloudCNN()
        
        # Cargar pesos
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Almacenar activaciones
        self.activations = {}
        self._register_hooks()
        
        print(f"✓ Modelo cargado desde: {model_path}")
        print(f"✓ Dispositivo: {self.device}")
    
    def _register_hooks(self):
        """Registra hooks para capturar activaciones de cada capa."""
        
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach().cpu()
            return hook
        
        # Registrar hooks en cada capa convolucional
        self.model.conv1.register_forward_hook(get_activation('Conv1 (3→32)'))
        self.model.conv2.register_forward_hook(get_activation('Conv2 (32→64)'))
        self.model.conv3.register_forward_hook(get_activation('Conv3 (64→128)'))
        self.model.conv4.register_forward_hook(get_activation('Conv4 (128→256)'))
    
    def visualize_image(self, image_path, save_path=None):
        """
        Visualiza los feature maps para una imagen.
        
        Args:
            image_path: Ruta a la imagen
            save_path: Ruta donde guardar la visualización (opcional)
        """
        # Limpiar activaciones previas
        self.activations.clear()
        
        # Cargar y procesar imagen
        image_tensor = get_single_image_loader(image_path).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            output = self.model(image_tensor)
            probs = F.softmax(output, dim=1)
            conf, pred_idx = probs.max(1)
        
        pred_class = CLASS_NAMES[pred_idx.item()]
        pred_name = CLOUD_CLASSES.get(pred_class, pred_class)
        confidence = conf.item() * 100
        
        # Cargar imagen original
        original_img = Image.open(image_path).convert('RGB')
        
        # Crear figura
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle(f'Análisis de Filtros CNN\n{Path(image_path).name}\nPredicción: {pred_class} ({pred_name}) - {confidence:.1f}%', 
                     fontsize=14, fontweight='bold')
        
        # 1. Imagen original
        ax_orig = fig.add_subplot(5, 1, 1)
        ax_orig.imshow(original_img)
        ax_orig.set_title('Imagen Original', fontsize=12)
        ax_orig.axis('off')
        
        # 2-5. Feature maps de cada capa
        layer_names = list(self.activations.keys())
        
        for layer_idx, layer_name in enumerate(layer_names):
            activation = self.activations[layer_name][0]  # [C, H, W]
            num_filters = activation.shape[0]
            
            # Mostrar los primeros 8 filtros más activos
            filter_activations = activation.mean(dim=(1, 2))  # Promedio por filtro
            top_filters = torch.argsort(filter_activations, descending=True)[:8]
            
            # Crear subplot para esta capa
            for i, filter_idx in enumerate(top_filters):
                ax = fig.add_subplot(5, 8, (layer_idx + 1) * 8 + i + 1)
                
                feature_map = activation[filter_idx].numpy()
                
                # Normalizar para visualización
                feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
                
                ax.imshow(feature_map, cmap='viridis')
                if i == 0:
                    ax.set_ylabel(layer_name, fontsize=10, rotation=0, labelpad=60, va='center')
                ax.set_title(f'F{filter_idx.item()}', fontsize=8)
                ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Guardado: {save_path}")
        
        plt.show()
        plt.close()
        
        return pred_class, confidence
    
    def visualize_all_classes(self, output_dir='filter_visualizations'):
        """
        Visualiza una imagen de cada clase.
        
        Args:
            output_dir: Carpeta donde guardar las visualizaciones
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("\n" + "=" * 60)
        print("VISUALIZACIÓN DE FILTROS POR CLASE")
        print("=" * 60)
        
        results = []
        
        for class_code in CLASS_NAMES:
            class_dir = DATA_DIR / class_code
            if not class_dir.exists():
                print(f"⚠ Carpeta no encontrada: {class_dir}")
                continue
            
            # Buscar primera imagen
            extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif']
            image_files = []
            for ext in extensions:
                image_files.extend(class_dir.glob(f'*{ext}'))
                image_files.extend(class_dir.glob(f'*{ext.upper()}'))
            
            if not image_files:
                print(f"⚠ Sin imágenes en: {class_dir}")
                continue
            
            image_path = image_files[0]
            save_path = output_path / f'{class_code}_filters.png'
            
            print(f"\n📷 Procesando: {class_code} ({CLOUD_CLASSES.get(class_code, '')})")
            print(f"   Imagen: {image_path.name}")
            
            pred_class, confidence = self.visualize_image(str(image_path), str(save_path))
            
            status = "✓" if pred_class == class_code else "✗"
            results.append((class_code, pred_class, confidence, status))
        
        # Resumen
        print("\n" + "=" * 60)
        print("RESUMEN DE PREDICCIONES")
        print("=" * 60)
        print(f"{'Clase Real':<10} {'Predicción':<10} {'Confianza':<12} {'Estado'}")
        print("-" * 45)
        
        correct = 0
        for real, pred, conf, status in results:
            print(f"{real:<10} {pred:<10} {conf:>8.1f}%    {status}")
            if status == "✓":
                correct += 1
        
        print("-" * 45)
        print(f"Accuracy: {correct}/{len(results)} ({100*correct/len(results):.1f}%)")
        print(f"\n✓ Visualizaciones guardadas en: {output_path.absolute()}")


def visualize_single_filter_details(image_path, output_path='filter_detail.png'):
    """
    Visualización detallada de TODOS los filtros de cada capa para una imagen.
    """
    visualizer = FilterVisualizer()
    
    # Procesar imagen
    image_tensor = get_single_image_loader(image_path).to(visualizer.device)
    
    with torch.no_grad():
        output = visualizer.model(image_tensor)
        probs = F.softmax(output, dim=1)
        conf, pred_idx = probs.max(1)
    
    pred_class = CLASS_NAMES[pred_idx.item()]
    
    # Crear figura grande con todos los filtros
    fig, axes = plt.subplots(4, 1, figsize=(20, 24))
    fig.suptitle(f'Detalle de Filtros: {Path(image_path).name}\nPredicción: {pred_class} ({conf.item()*100:.1f}%)', 
                 fontsize=16, fontweight='bold')
    
    layer_names = list(visualizer.activations.keys())
    
    for layer_idx, (ax, layer_name) in enumerate(zip(axes, layer_names)):
        activation = visualizer.activations[layer_name][0]  # [C, H, W]
        num_filters = activation.shape[0]
        
        # Crear mosaico de todos los filtros
        cols = 16 if num_filters >= 16 else num_filters
        rows = (num_filters + cols - 1) // cols
        
        mosaic = np.zeros((rows * activation.shape[1], cols * activation.shape[2]))
        
        for i in range(num_filters):
            row = i // cols
            col = i % cols
            feature_map = activation[i].numpy()
            feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
            
            mosaic[row*activation.shape[1]:(row+1)*activation.shape[1],
                   col*activation.shape[2]:(col+1)*activation.shape[2]] = feature_map
        
        ax.imshow(mosaic, cmap='viridis')
        ax.set_title(f'{layer_name} - {num_filters} filtros (tamaño: {activation.shape[1]}x{activation.shape[2]})', 
                     fontsize=12)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Guardado: {output_path}")
    plt.show()
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("VISUALIZADOR DE FILTROS CNN - CloudClassify13")
    print("=" * 60)
    
    visualizer = FilterVisualizer()
    
    # Visualizar una imagen de cada clase
    visualizer.visualize_all_classes()
    
    print("\n" + "=" * 60)
    print("¡Visualización completada!")
    print("Las imágenes muestran qué 've' cada capa convolucional.")
    print("=" * 60)
