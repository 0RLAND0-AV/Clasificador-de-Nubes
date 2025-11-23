"""
CloudClassify13 - Predicción e Inferencia
==========================================
Funciones para realizar predicciones sobre nuevas imágenes.

Archivo: predict.py
Descripción: Implementa funciones de inferencia para clasificar
             imágenes de nubes usando el modelo entrenado.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image

from config import (
    CLASS_NAMES, CLOUD_CLASSES, MODEL_PATH, IMAGE_SIZE,
    NORMALIZE_MEAN, NORMALIZE_STD, MIN_CONFIDENCE, TOP_K
)
from dataset import get_single_image_loader
from model import create_model


class CloudPredictor:
    """
    Clase para realizar predicciones sobre imágenes de nubes.
    """
    
    def __init__(self, model_path=MODEL_PATH, device='cpu'):
        """
        Inicializa el predictor.
        
        Args:
            model_path: Ruta al modelo entrenado
            device: Dispositivo ('cpu' o 'cuda')
        """
        self.device = device
        self.model_path = model_path
        
        # Cargar modelo
        self.model = create_model(device=device)
        self._load_model(model_path)
        self.model.eval()
        
        print(f"✓ Modelo cargado desde: {model_path}")
        print(f"✓ Dispositivo: {device}")
    
    def _load_model(self, model_path):
        """Carga los pesos del modelo."""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Archivo de modelo no encontrado: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Soportar tanto archivos .pth simples como checkpoints completos
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
    
    def predict_single(self, image_path, return_probabilities=False):
        """
        Realiza una predicción sobre una imagen individual.
        
        Args:
            image_path: Ruta a la imagen
            return_probabilities: Si es True, retorna probabilidades
        
        Returns:
            dict: Predicción con clase, confianza y probabilidades (opcional)
        """
        # Cargar y procesar imagen
        image_tensor = get_single_image_loader(image_path)
        image_tensor = image_tensor.to(self.device)
        
        # Predicción
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_idx = probabilities.max(1)
        
        predicted_class = CLASS_NAMES[predicted_idx.item()]
        confidence_score = confidence.item()
        
        result = {
            'image_path': str(image_path),
            'predicted_class': predicted_class,
            'predicted_class_name': CLOUD_CLASSES.get(predicted_class, predicted_class),
            'confidence': confidence_score,
            'confidence_percent': f"{confidence_score * 100:.2f}%"
        }
        
        if return_probabilities:
            # Obtener top-K predicciones
            top_probs, top_indices = probabilities.topk(TOP_K, dim=1)
            
            top_predictions = []
            for prob, idx in zip(top_probs[0], top_indices[0]):
                class_code = CLASS_NAMES[idx.item()]
                top_predictions.append({
                    'class': class_code,
                    'class_name': CLOUD_CLASSES.get(class_code, class_code),
                    'probability': prob.item(),
                    'probability_percent': f"{prob.item() * 100:.2f}%"
                })
            
            result['all_probabilities'] = {CLASS_NAMES[i]: p.item() for i, p in enumerate(probabilities[0])}
            result['top_predictions'] = top_predictions
        
        return result
    
    def predict_batch(self, image_paths, return_probabilities=False):
        """
        Realiza predicciones sobre múltiples imágenes.
        
        Args:
            image_paths: Lista de rutas a imágenes
            return_probabilities: Si es True, retorna probabilidades
        
        Returns:
            list: Lista de predicciones
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.predict_single(image_path, return_probabilities)
                results.append(result)
            except Exception as e:
                results.append({
                    'image_path': str(image_path),
                    'error': str(e)
                })
        
        return results
    
    def predict_with_visualization(self, image_path):
        """
        Realiza predicción y retorna información para visualización.
        
        Args:
            image_path: Ruta a la imagen
        
        Returns:
            dict: Información de predicción y visualización
        """
        result = self.predict_single(image_path, return_probabilities=True)
        
        # Cargar imagen original para visualización
        image = Image.open(image_path)
        
        result['image'] = image
        result['image_size'] = image.size
        
        return result


def predict_image(image_path, model_path=MODEL_PATH, device='cpu', verbose=True):
    """
    Función simple para hacer una predicción.
    
    Args:
        image_path: Ruta a la imagen
        model_path: Ruta al modelo
        device: Dispositivo ('cpu' o 'cuda')
        verbose: Si es True, imprime resultados
    
    Returns:
        dict: Resultado de predicción
    """
    predictor = CloudPredictor(model_path=model_path, device=device)
    result = predictor.predict_single(image_path, return_probabilities=True)
    
    if verbose:
        print("\n" + "=" * 60)
        print("PREDICCIÓN DE TIPO DE NUBE")
        print("=" * 60)
        print(f"Imagen: {result['image_path']}")
        print(f"Predicción: {result['predicted_class_name']}")
        print(f"Confianza: {result['confidence_percent']}")
        print("\nTop-3 Predicciones:")
        for i, pred in enumerate(result['top_predictions'], 1):
            print(f"  {i}. {pred['class_name']:20s} - {pred['probability_percent']}")
        print("=" * 60 + "\n")
    
    return result


if __name__ == "__main__":
    print("Módulo de predicción importado correctamente ✓")
