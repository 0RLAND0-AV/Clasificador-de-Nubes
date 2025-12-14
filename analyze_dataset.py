"""
CloudClassify13 - Análisis y Auditoría del Dataset
===================================================
Identifica imágenes potencialmente mal clasificadas basándose en:
1. Predicciones del modelo CNN
2. Análisis visual teórico de características de nubes

Archivo: analyze_dataset.py
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import os
from collections import defaultdict

from config import MODEL_PATH, CLASS_NAMES, CLOUD_CLASSES, DATA_DIR, IMAGE_SIZE
from dataset import get_single_image_loader
from model import CloudCNN


# ==================== DEFINICIONES TEÓRICAS DE NUBES ====================
CLOUD_DEFINITIONS = {
    'Cu': {
        'name': 'Cúmulo',
        'description': 'Proyección ascendente circular; nubes tamaño puño; márgenes claros',
        'characteristics': {
            'edges': 'sharp',           # Bordes definidos
            'shape': 'rounded',         # Forma redondeada
            'color': 'white',           # Blanco
            'coverage': 'partial',      # Cobertura parcial
            'texture': 'puffy'          # Textura esponjosa
        }
    },
    'Cb': {
        'name': 'Cumulonimbo',
        'description': 'Nubes gruesas como brócoli; bordes borrosos',
        'characteristics': {
            'edges': 'fuzzy',
            'shape': 'towering',
            'color': 'dark_gray',
            'coverage': 'variable',
            'texture': 'dense'
        }
    },
    'Sc': {
        'name': 'Estratocúmulo',
        'description': 'Nubes tamaño puño distribuidas, agrupadas, onduladas; grises o blanco grisáceo',
        'characteristics': {
            'edges': 'moderate',
            'shape': 'patchy',
            'color': 'gray_white',
            'coverage': 'partial',
            'texture': 'wavy'
        }
    },
    'St': {
        'name': 'Estrato',
        'description': 'Nubes uniformes; cubren gran área/todo el cielo; mayormente grises',
        'characteristics': {
            'edges': 'none',            # Sin bordes definidos
            'shape': 'layer',           # Capa uniforme
            'color': 'gray',            # Gris
            'coverage': 'full',         # Cobertura total
            'texture': 'uniform'        # Textura uniforme
        }
    },
    'Ns': {
        'name': 'Nimboestrato',
        'description': 'Nubes bajas amorfas; oscurecen sol/luna; esponjosas gris oscuro',
        'characteristics': {
            'edges': 'none',
            'shape': 'amorphous',
            'color': 'dark_gray',
            'coverage': 'full',
            'texture': 'fluffy_dark'
        }
    },
    'As': {
        'name': 'Altoestrato',
        'description': 'Nubes gruesas cubriendo cielo; sol casi no atraviesa; estructuras rayadas; blanco/azul grisáceo',
        'characteristics': {
            'edges': 'soft',
            'shape': 'layer',
            'color': 'gray_blue',
            'coverage': 'full',
            'texture': 'striated'       # Rayado
        }
    },
    'Ac': {
        'name': 'Altocúmulo',
        'description': 'Nubes pequeñas contorno definido; ovaladas, forma teja; distribución ondulada escamas pez',
        'characteristics': {
            'edges': 'defined',
            'shape': 'oval_tiles',
            'color': 'white_gray',
            'coverage': 'partial',
            'texture': 'fish_scales'    # Escamas de pez
        }
    },
    'Ci': {
        'name': 'Cirros',
        'description': 'Delgadas transparentes; blancas brillantes; filamentosas como colas de caballo',
        'characteristics': {
            'edges': 'wispy',
            'shape': 'filaments',
            'color': 'white_bright',
            'coverage': 'sparse',
            'texture': 'feathery'
        }
    },
    'Cs': {
        'name': 'Cirroestrato',
        'description': 'Base filamentosa; lo suficientemente delgado para ver sol/luna; HALO distintivo',
        'characteristics': {
            'edges': 'soft',
            'shape': 'veil',
            'color': 'white_translucent',
            'coverage': 'full',
            'texture': 'thin_veil',
            'special': 'halo'           # Característica especial: HALO SOLAR
        }
    },
    'Cc': {
        'name': 'Cirrocúmulo',
        'description': 'Nubes MUY pequeñas, blancas brillantes; escamas blancas delgadas; en filas y grupos',
        'characteristics': {
            'edges': 'tiny_defined',
            'shape': 'tiny_puffs',
            'color': 'white_bright',
            'coverage': 'partial',
            'texture': 'rippled'        # Ondulado pequeño
        }
    },
    'Ct': {
        'name': 'Contrails',
        'description': 'Estelas de aviones; líneas rectas en el cielo',
        'characteristics': {
            'edges': 'linear',
            'shape': 'lines',
            'color': 'white',
            'coverage': 'minimal',
            'texture': 'linear'
        }
    },
    'Nc': {
        'name': 'Sin Nubes',
        'description': 'No hay nubes; fondo azul; quizás solo el sol',
        'characteristics': {
            'edges': 'none',
            'shape': 'none',
            'color': 'blue',
            'coverage': 'none',
            'texture': 'clear'
        }
    }
}


class DatasetAnalyzer:
    """Analiza el dataset para identificar imágenes problemáticas."""
    
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
        
        print(f"✓ Modelo cargado desde: {model_path}")
        print(f"✓ Dispositivo: {self.device}")
        
        # Clases que requieren análisis especial
        self.special_classes = ['Ac', 'Cc', 'Ns', 'St', 'As']
        
        # Resultados
        self.results = defaultdict(list)
        self.problematic_images = defaultdict(list)
    
    def get_predictions(self, image_path):
        """Obtiene predicciones para una imagen."""
        image_tensor = get_single_image_loader(image_path).to(self.device)
        
        with torch.no_grad():
            output = self.model(image_tensor)
            probs = F.softmax(output, dim=1)[0]
        
        # Ordenar por probabilidad
        sorted_indices = torch.argsort(probs, descending=True)
        
        predictions = []
        for idx in sorted_indices:
            class_code = CLASS_NAMES[idx.item()]
            prob = probs[idx].item() * 100
            predictions.append({
                'class': class_code,
                'name': CLOUD_CLASSES.get(class_code, class_code),
                'probability': prob
            })
        
        return predictions
    
    def analyze_image_characteristics(self, image_path):
        """
        Analiza características visuales de la imagen.
        Retorna métricas que ayudan a identificar el tipo de nube.
        """
        # Cargar imagen
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        characteristics = {}
        
        # 1. Análisis de color
        h, s, v = cv2.split(img_hsv)
        b, g, r = cv2.split(img_rgb)
        
        characteristics['brightness_mean'] = np.mean(v)
        characteristics['brightness_std'] = np.std(v)
        characteristics['saturation_mean'] = np.mean(s)
        
        # Detectar si es mayormente azul (cielo despejado)
        blue_ratio = np.mean(b) / (np.mean(r) + np.mean(g) + np.mean(b) + 1e-6)
        characteristics['blue_ratio'] = blue_ratio
        
        # Detectar si es mayormente gris
        gray_diff = np.std([np.mean(r), np.mean(g), np.mean(b)])
        characteristics['is_gray'] = gray_diff < 15  # Poco diferencia entre canales = gris
        characteristics['gray_level'] = np.mean(img_gray)
        
        # 2. Análisis de bordes (para detectar nubes con contornos definidos)
        edges = cv2.Canny(img_gray, 50, 150)
        characteristics['edge_density'] = np.sum(edges > 0) / edges.size
        
        # 3. Análisis de textura usando Laplaciano (varianza)
        laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
        characteristics['texture_variance'] = laplacian.var()
        
        # 4. Detección de líneas (para Contrails y Ci)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
        characteristics['line_count'] = len(lines) if lines is not None else 0
        
        # 5. Análisis de cobertura del cielo
        # Umbral para separar nubes (brillante) del cielo (azul)
        _, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
        characteristics['coverage'] = np.sum(thresh > 0) / thresh.size
        
        # 6. Detectar uniformidad (para St, Ns)
        # Dividir imagen en cuadrantes y comparar
        h_img, w_img = img_gray.shape
        quadrants = [
            img_gray[:h_img//2, :w_img//2],
            img_gray[:h_img//2, w_img//2:],
            img_gray[h_img//2:, :w_img//2],
            img_gray[h_img//2:, w_img//2:]
        ]
        quad_means = [np.mean(q) for q in quadrants]
        characteristics['uniformity'] = 1 - (np.std(quad_means) / (np.mean(quad_means) + 1e-6))
        
        # 7. Detectar pequeños elementos repetitivos (para Cc, Ac)
        # Usando detección de blobs
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 50
        params.maxArea = 5000
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(255 - img_gray)  # Invertir para detectar nubes blancas
        characteristics['blob_count'] = len(keypoints)
        
        # 8. Detectar halo (para Cs) - buscar círculo brillante
        # Detectar círculos
        circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, 100,
                                   param1=50, param2=30, minRadius=30, maxRadius=200)
        characteristics['has_halo'] = circles is not None and len(circles) > 0
        
        return characteristics
    
    def evaluate_theoretical_match(self, class_code, characteristics):
        """
        Evalúa qué tan bien coincide la imagen con la definición teórica de la clase.
        Retorna un puntaje de 0-100.
        """
        if class_code not in CLOUD_DEFINITIONS or characteristics is None:
            return 50  # Neutral
        
        definition = CLOUD_DEFINITIONS[class_code]['characteristics']
        score = 50  # Base
        reasons = []
        
        # Evaluar según la clase
        if class_code == 'St':
            # Estrato: uniforme, gris, cobertura total
            if characteristics['uniformity'] > 0.7:
                score += 15
                reasons.append("Alta uniformidad ✓")
            else:
                score -= 15
                reasons.append("Baja uniformidad ✗")
            
            if characteristics['is_gray']:
                score += 10
                reasons.append("Color gris ✓")
            
            if characteristics['coverage'] > 0.7:
                score += 10
                reasons.append("Alta cobertura ✓")
            
            if characteristics['edge_density'] < 0.05:
                score += 10
                reasons.append("Pocos bordes (uniforme) ✓")
        
        elif class_code == 'As':
            # Altoestrato: capas, gris-azul, cobertura alta
            if characteristics['uniformity'] > 0.5:
                score += 10
                reasons.append("Uniformidad moderada ✓")
            
            if 100 < characteristics['gray_level'] < 180:
                score += 10
                reasons.append("Nivel gris medio ✓")
            
            if characteristics['texture_variance'] > 100:
                score += 10
                reasons.append("Textura rayada ✓")
        
        elif class_code == 'Ns':
            # Nimboestrato: oscuro, amorfo, cobertura total
            if characteristics['gray_level'] < 120:
                score += 15
                reasons.append("Oscuro ✓")
            else:
                score -= 10
                reasons.append("No es oscuro ✗")
            
            if characteristics['uniformity'] > 0.6:
                score += 10
                reasons.append("Amorfo uniforme ✓")
            
            if characteristics['coverage'] > 0.8:
                score += 10
                reasons.append("Cobertura total ✓")
        
        elif class_code == 'Ac':
            # Altocúmulo: pequeños elementos, contornos, escamas de pez
            if characteristics['blob_count'] > 10:
                score += 20
                reasons.append(f"Muchos elementos pequeños ({characteristics['blob_count']}) ✓")
            elif characteristics['blob_count'] > 5:
                score += 10
                reasons.append(f"Algunos elementos ({characteristics['blob_count']}) ✓")
            
            if characteristics['edge_density'] > 0.05:
                score += 10
                reasons.append("Bordes definidos ✓")
        
        elif class_code == 'Cc':
            # Cirrocúmulo: MUY pequeños, brillantes, en filas
            if characteristics['brightness_mean'] > 150:
                score += 15
                reasons.append("Brillante ✓")
            
            if characteristics['blob_count'] > 20:
                score += 20
                reasons.append(f"Muchos elementos muy pequeños ({characteristics['blob_count']}) ✓")
        
        elif class_code == 'Cs':
            # Cirroestrato: delgado, translúcido, HALO
            if characteristics['has_halo']:
                score += 30
                reasons.append("¡HALO DETECTADO! ✓")
            
            if characteristics['brightness_mean'] > 160:
                score += 10
                reasons.append("Muy brillante/translúcido ✓")
            
            if characteristics['uniformity'] > 0.6:
                score += 10
                reasons.append("Velo uniforme ✓")
        
        elif class_code == 'Ci':
            # Cirros: filamentos, líneas, brillante
            if characteristics['line_count'] > 5:
                score += 15
                reasons.append(f"Líneas/filamentos detectados ({characteristics['line_count']}) ✓")
            
            if characteristics['brightness_mean'] > 170:
                score += 10
                reasons.append("Brillante ✓")
        
        elif class_code == 'Ct':
            # Contrails: líneas rectas
            if characteristics['line_count'] > 3:
                score += 25
                reasons.append(f"Líneas rectas ({characteristics['line_count']}) ✓")
        
        elif class_code == 'Nc':
            # Sin nubes: azul, sin nubes
            if characteristics['blue_ratio'] > 0.4:
                score += 25
                reasons.append("Predomina azul ✓")
            
            if characteristics['coverage'] < 0.3:
                score += 15
                reasons.append("Poca cobertura ✓")
        
        elif class_code == 'Cu':
            # Cúmulo: bordes definidos, forma redondeada
            if characteristics['edge_density'] > 0.08:
                score += 15
                reasons.append("Bordes muy definidos ✓")
            
            if 0.3 < characteristics['coverage'] < 0.7:
                score += 10
                reasons.append("Cobertura parcial ✓")
        
        elif class_code == 'Cb':
            # Cumulonimbo: grande, oscuro, bordes difusos
            if characteristics['gray_level'] < 140:
                score += 10
                reasons.append("Oscuro ✓")
            
            if characteristics['texture_variance'] > 500:
                score += 15
                reasons.append("Alta variación de textura ✓")
        
        elif class_code == 'Sc':
            # Estratocúmulo: parches, ondulado
            if 5 < characteristics['blob_count'] < 30:
                score += 15
                reasons.append(f"Parches detectados ({characteristics['blob_count']}) ✓")
            
            if 0.1 < characteristics['edge_density'] < 0.15:
                score += 10
                reasons.append("Bordes moderados ✓")
        
        return min(100, max(0, score)), reasons
    
    def analyze_class(self, class_code):
        """Analiza todas las imágenes de una clase."""
        class_dir = DATA_DIR / class_code
        
        if not class_dir.exists():
            print(f"⚠ Carpeta no encontrada: {class_dir}")
            return
        
        # Buscar imágenes
        extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif']
        image_files = []
        for ext in extensions:
            image_files.extend(class_dir.glob(f'*{ext}'))
            image_files.extend(class_dir.glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"⚠ Sin imágenes en: {class_dir}")
            return
        
        print(f"\n{'='*70}")
        print(f"📂 ANALIZANDO: {class_code} ({CLOUD_CLASSES.get(class_code, '')}) - {len(image_files)} imágenes")
        print(f"{'='*70}")
        
        if class_code in CLOUD_DEFINITIONS:
            print(f"📖 Definición: {CLOUD_DEFINITIONS[class_code]['description']}")
        print()
        
        correct_count = 0
        problematic = []
        
        for img_path in sorted(image_files):
            predictions = self.get_predictions(str(img_path))
            top_pred = predictions[0]
            second_pred = predictions[1]
            
            # Análisis visual (solo para clases especiales)
            theory_score = 50
            theory_reasons = []
            if class_code in self.special_classes:
                characteristics = self.analyze_image_characteristics(str(img_path))
                if characteristics:
                    theory_score, theory_reasons = self.evaluate_theoretical_match(class_code, characteristics)
            
            # Determinar si es problemática
            is_correct = top_pred['class'] == class_code
            is_low_confidence = top_pred['probability'] < 50
            is_confused = (not is_correct) or (second_pred['probability'] > 30 and top_pred['class'] != class_code)
            theory_mismatch = theory_score < 40
            
            is_problematic = (not is_correct) or is_low_confidence or theory_mismatch
            
            if is_correct:
                correct_count += 1
            
            # Guardar resultado
            result = {
                'image': img_path.name,
                'path': str(img_path),
                'expected': class_code,
                'predicted': top_pred['class'],
                'confidence': top_pred['probability'],
                'second': second_pred['class'],
                'second_conf': second_pred['probability'],
                'all_predictions': predictions[:3],
                'is_correct': is_correct,
                'theory_score': theory_score,
                'theory_reasons': theory_reasons
            }
            
            self.results[class_code].append(result)
            
            if is_problematic:
                problematic.append(result)
                self.problematic_images[class_code].append(result)
                
                # Imprimir detalle
                status = "✗" if not is_correct else "⚠"
                print(f"{status} {img_path.name}")
                print(f"   Predicción: {top_pred['class']} ({top_pred['probability']:.1f}%) vs Esperado: {class_code}")
                print(f"   Top 3: {top_pred['class']} {top_pred['probability']:.1f}% | {second_pred['class']} {second_pred['probability']:.1f}% | {predictions[2]['class']} {predictions[2]['probability']:.1f}%")
                
                if class_code in self.special_classes:
                    print(f"   Análisis Teórico: {theory_score}/100")
                    for reason in theory_reasons:
                        print(f"      - {reason}")
                
                print()
        
        # Resumen de la clase
        accuracy = (correct_count / len(image_files)) * 100
        print(f"📊 Resumen {class_code}: {correct_count}/{len(image_files)} correctas ({accuracy:.1f}%)")
        print(f"   Problemáticas identificadas: {len(problematic)}")
        
        return {
            'total': len(image_files),
            'correct': correct_count,
            'problematic': len(problematic),
            'accuracy': accuracy
        }
    
    def analyze_all(self):
        """Analiza todo el dataset."""
        print("\n" + "=" * 70)
        print("🔍 ANÁLISIS COMPLETO DEL DATASET - CloudClassify13")
        print("=" * 70)
        print("Este análisis identifica imágenes potencialmente mal clasificadas")
        print("basándose en predicciones del modelo y análisis teórico de nubes.")
        print("=" * 70)
        
        class_stats = {}
        
        for class_code in CLASS_NAMES:
            stats = self.analyze_class(class_code)
            if stats:
                class_stats[class_code] = stats
        
        # Resumen final
        self.print_final_summary(class_stats)
    
    def print_final_summary(self, class_stats):
        """Imprime resumen final del análisis."""
        print("\n")
        print("=" * 70)
        print("📋 RESUMEN FINAL DEL ANÁLISIS")
        print("=" * 70)
        
        total_images = sum(s['total'] for s in class_stats.values())
        total_correct = sum(s['correct'] for s in class_stats.values())
        total_problematic = sum(s['problematic'] for s in class_stats.values())
        
        print(f"\n{'Clase':<8} {'Total':<8} {'Correctas':<12} {'Problemáticas':<15} {'Accuracy'}")
        print("-" * 55)
        
        for class_code in CLASS_NAMES:
            if class_code in class_stats:
                s = class_stats[class_code]
                special = "⭐" if class_code in self.special_classes else ""
                print(f"{class_code:<8} {s['total']:<8} {s['correct']:<12} {s['problematic']:<15} {s['accuracy']:>6.1f}% {special}")
        
        print("-" * 55)
        print(f"{'TOTAL':<8} {total_images:<8} {total_correct:<12} {total_problematic:<15} {100*total_correct/total_images:>6.1f}%")
        print()
        print("⭐ = Clases con análisis teórico especial (Ac, Cc, Ns, St, As)")
        
        # Listar imágenes problemáticas por clase
        print("\n")
        print("=" * 70)
        print("🚨 LISTADO DE IMÁGENES PROBLEMÁTICAS")
        print("=" * 70)
        
        for class_code in CLASS_NAMES:
            if class_code in self.problematic_images and self.problematic_images[class_code]:
                print(f"\n📁 {class_code} ({CLOUD_CLASSES.get(class_code, '')}):")
                print("-" * 50)
                
                for img in self.problematic_images[class_code]:
                    pred = img['predicted']
                    conf = img['confidence']
                    second = img['second']
                    second_conf = img['second_conf']
                    
                    if img['is_correct']:
                        issue = f"Baja confianza ({conf:.1f}%)"
                    else:
                        issue = f"Predicho como {pred} ({conf:.1f}%)"
                    
                    print(f"  • {img['image']}")
                    print(f"    Problema: {issue}")
                    print(f"    Podría ser: {pred} ({conf:.1f}%) o {second} ({second_conf:.1f}%)")
                    if img['theory_score'] < 50 and class_code in self.special_classes:
                        print(f"    Análisis teórico: {img['theory_score']}/100 (bajo)")
        
        print("\n")
        print("=" * 70)
        print("✅ ANÁLISIS COMPLETADO")
        print("=" * 70)
        print("Las imágenes listadas arriba podrían necesitar revisión manual.")
        print("Considera moverlas a la clase correcta o eliminarlas del dataset.")
        print("=" * 70)


if __name__ == "__main__":
    analyzer = DatasetAnalyzer()
    analyzer.analyze_all()
