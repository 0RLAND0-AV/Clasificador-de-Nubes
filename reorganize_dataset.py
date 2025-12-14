"""
CloudClassify13 - Reorganizar Dataset
=====================================
Mueve imágenes mal clasificadas a su carpeta correcta según predicciones.
Permite revertir los cambios.

Archivo: reorganize_dataset.py
"""

import torch
import torch.nn.functional as F
import json
import shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from config import MODEL_PATH, CLASS_NAMES, CLOUD_CLASSES, DATA_DIR
from dataset import get_single_image_loader
from model import CloudCNN


# Archivo para guardar historial de movimientos (para revertir)
HISTORY_FILE = Path(__file__).parent / "dataset_changes_history.json"


class DatasetReorganizer:
    """Reorganiza el dataset basándose en predicciones del modelo."""
    
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
        
        print(f"✓ Modelo cargado")
        print(f"✓ Dispositivo: {self.device}")
        
        # Cargar historial si existe
        self.history = self._load_history()
    
    def _load_history(self):
        """Carga historial de cambios."""
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"sessions": []}
    
    def _save_history(self):
        """Guarda historial de cambios."""
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
    
    def get_prediction(self, image_path):
        """Obtiene predicción para una imagen."""
        image_tensor = get_single_image_loader(image_path).to(self.device)
        
        with torch.no_grad():
            output = self.model(image_tensor)
            probs = F.softmax(output, dim=1)[0]
        
        sorted_indices = torch.argsort(probs, descending=True)
        
        top_class = CLASS_NAMES[sorted_indices[0].item()]
        top_prob = probs[sorted_indices[0]].item() * 100
        second_class = CLASS_NAMES[sorted_indices[1].item()]
        second_prob = probs[sorted_indices[1]].item() * 100
        
        return {
            'top_class': top_class,
            'top_prob': top_prob,
            'second_class': second_class,
            'second_prob': second_prob
        }
    
    def analyze_and_suggest(self, min_confidence=60, ct_delete_threshold=75):
        """
        Analiza el dataset y genera sugerencias de movimiento.
        
        Args:
            min_confidence: Mínimo % de confianza para mover (default: 60%)
            ct_delete_threshold: Para Ct, eliminar si otra clase tiene >75%
        """
        print("\n" + "=" * 70)
        print("📊 ANALIZANDO DATASET PARA REORGANIZACIÓN")
        print(f"   Umbral de confianza para mover: {min_confidence}%")
        print(f"   Umbral para eliminar Contrails: {ct_delete_threshold}%")
        print("=" * 70)
        
        suggestions = {
            'move': [],      # Imágenes a mover a otra clase
            'delete': [],    # Imágenes a eliminar (solo Ct problemáticos)
            'keep': []       # Imágenes que se quedan
        }
        
        extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif']
        
        for class_code in CLASS_NAMES:
            class_dir = DATA_DIR / class_code
            if not class_dir.exists():
                continue
            
            # Buscar imágenes
            image_files = []
            for ext in extensions:
                image_files.extend(class_dir.glob(f'*{ext}'))
                image_files.extend(class_dir.glob(f'*{ext.upper()}'))
            
            for img_path in sorted(image_files):
                pred = self.get_prediction(str(img_path))
                
                is_correct = pred['top_class'] == class_code
                
                if is_correct:
                    suggestions['keep'].append({
                        'path': str(img_path),
                        'original_class': class_code,
                        'predicted_class': pred['top_class'],
                        'confidence': pred['top_prob']
                    })
                else:
                    # Caso especial para Contrails
                    if class_code == 'Ct':
                        if pred['top_prob'] >= ct_delete_threshold:
                            suggestions['delete'].append({
                                'path': str(img_path),
                                'original_class': class_code,
                                'predicted_class': pred['top_class'],
                                'confidence': pred['top_prob'],
                                'reason': f"Ct predicho como {pred['top_class']} con {pred['top_prob']:.1f}%"
                            })
                        else:
                            suggestions['keep'].append({
                                'path': str(img_path),
                                'original_class': class_code,
                                'predicted_class': pred['top_class'],
                                'confidence': pred['top_prob'],
                                'note': 'Ct mantenido (confianza < umbral)'
                            })
                    else:
                        # Para otras clases: mover si confianza >= umbral
                        if pred['top_prob'] >= min_confidence:
                            suggestions['move'].append({
                                'path': str(img_path),
                                'original_class': class_code,
                                'target_class': pred['top_class'],
                                'confidence': pred['top_prob']
                            })
                        else:
                            suggestions['keep'].append({
                                'path': str(img_path),
                                'original_class': class_code,
                                'predicted_class': pred['top_class'],
                                'confidence': pred['top_prob'],
                                'note': 'Confianza insuficiente para mover'
                            })
        
        return suggestions
    
    def print_suggestions(self, suggestions):
        """Imprime las sugerencias de manera legible."""
        print("\n" + "=" * 70)
        print("📋 SUGERENCIAS DE REORGANIZACIÓN")
        print("=" * 70)
        
        # Imágenes a mover
        print(f"\n🔄 IMÁGENES A MOVER ({len(suggestions['move'])} archivos):")
        print("-" * 60)
        
        by_move = defaultdict(list)
        for item in suggestions['move']:
            key = f"{item['original_class']} → {item['target_class']}"
            by_move[key].append(item)
        
        for move_type, items in sorted(by_move.items()):
            print(f"\n  {move_type}:")
            for item in items:
                filename = Path(item['path']).name
                print(f"    • {filename} ({item['confidence']:.1f}%)")
        
        # Imágenes a eliminar (Ct)
        print(f"\n🗑️ CONTRAILS A ELIMINAR ({len(suggestions['delete'])} archivos):")
        print("-" * 60)
        
        for item in suggestions['delete']:
            filename = Path(item['path']).name
            print(f"    • {filename} - {item['reason']}")
        
        # Resumen
        print("\n" + "=" * 70)
        print("📊 RESUMEN:")
        print(f"   Mover: {len(suggestions['move'])} imágenes")
        print(f"   Eliminar: {len(suggestions['delete'])} imágenes (Ct)")
        print(f"   Mantener: {len(suggestions['keep'])} imágenes")
        print("=" * 70)
        
        return suggestions
    
    def apply_changes(self, suggestions, dry_run=False):
        """
        Aplica los cambios sugeridos.
        
        Args:
            suggestions: Diccionario de sugerencias
            dry_run: Si True, solo muestra qué haría sin hacer cambios
        """
        if dry_run:
            print("\n⚠️ MODO DRY-RUN: No se harán cambios reales")
        
        session = {
            'timestamp': datetime.now().isoformat(),
            'moves': [],
            'deletes': []
        }
        
        # Crear carpeta de eliminados
        deleted_dir = DATA_DIR / '_deleted'
        if not dry_run:
            deleted_dir.mkdir(exist_ok=True)
        
        print("\n" + "=" * 70)
        print("🔧 APLICANDO CAMBIOS...")
        print("=" * 70)
        
        # Mover imágenes
        moved_count = 0
        for item in suggestions['move']:
            src = Path(item['path'])
            dst = DATA_DIR / item['target_class'] / src.name
            
            # Evitar sobrescribir
            if dst.exists():
                base = dst.stem
                ext = dst.suffix
                counter = 1
                while dst.exists():
                    dst = DATA_DIR / item['target_class'] / f"{base}_{counter}{ext}"
                    counter += 1
            
            print(f"  📁 {src.name}: {item['original_class']} → {item['target_class']}")
            
            if not dry_run:
                shutil.move(str(src), str(dst))
                session['moves'].append({
                    'original': str(src),
                    'new': str(dst),
                    'original_class': item['original_class'],
                    'target_class': item['target_class']
                })
            
            moved_count += 1
        
        # Eliminar Contrails problemáticos (mover a _deleted)
        deleted_count = 0
        for item in suggestions['delete']:
            src = Path(item['path'])
            dst = deleted_dir / src.name
            
            # Evitar sobrescribir
            if dst.exists():
                base = dst.stem
                ext = dst.suffix
                counter = 1
                while dst.exists():
                    dst = deleted_dir / f"{base}_{counter}{ext}"
                    counter += 1
            
            print(f"  🗑️ {src.name}: eliminado (movido a _deleted)")
            
            if not dry_run:
                shutil.move(str(src), str(dst))
                session['deletes'].append({
                    'original': str(src),
                    'deleted_to': str(dst)
                })
            
            deleted_count += 1
        
        # Guardar historial
        if not dry_run and (session['moves'] or session['deletes']):
            self.history['sessions'].append(session)
            self._save_history()
        
        print("\n" + "=" * 70)
        print("✅ CAMBIOS APLICADOS:" if not dry_run else "📋 CAMBIOS QUE SE APLICARÍAN:")
        print(f"   Movidas: {moved_count} imágenes")
        print(f"   Eliminadas: {deleted_count} imágenes")
        if not dry_run:
            print(f"   Historial guardado en: {HISTORY_FILE}")
        print("=" * 70)
    
    def revert_last_session(self):
        """Revierte la última sesión de cambios."""
        if not self.history['sessions']:
            print("❌ No hay cambios para revertir")
            return
        
        session = self.history['sessions'].pop()
        
        print("\n" + "=" * 70)
        print("⏪ REVIRTIENDO ÚLTIMA SESIÓN...")
        print(f"   Fecha: {session['timestamp']}")
        print("=" * 70)
        
        # Revertir movimientos
        reverted_moves = 0
        for move in session['moves']:
            src = Path(move['new'])
            dst = Path(move['original'])
            
            if src.exists():
                # Asegurar que el directorio destino existe
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
                print(f"  ↩️ {src.name}: {move['target_class']} → {move['original_class']}")
                reverted_moves += 1
            else:
                print(f"  ⚠️ No encontrado: {src}")
        
        # Restaurar eliminados
        restored_deletes = 0
        for delete in session['deletes']:
            src = Path(delete['deleted_to'])
            dst = Path(delete['original'])
            
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
                print(f"  ♻️ Restaurado: {dst.name}")
                restored_deletes += 1
            else:
                print(f"  ⚠️ No encontrado: {src}")
        
        # Guardar historial actualizado
        self._save_history()
        
        print("\n" + "=" * 70)
        print("✅ REVERSIÓN COMPLETADA:")
        print(f"   Movimientos revertidos: {reverted_moves}")
        print(f"   Eliminados restaurados: {restored_deletes}")
        print("=" * 70)
    
    def show_history(self):
        """Muestra el historial de cambios."""
        if not self.history['sessions']:
            print("📋 No hay historial de cambios")
            return
        
        print("\n" + "=" * 70)
        print("📋 HISTORIAL DE CAMBIOS")
        print("=" * 70)
        
        for i, session in enumerate(self.history['sessions'], 1):
            print(f"\n  Sesión {i}: {session['timestamp']}")
            print(f"    Movimientos: {len(session['moves'])}")
            print(f"    Eliminaciones: {len(session['deletes'])}")


def main():
    """Función principal con menú interactivo."""
    import sys
    
    print("\n" + "=" * 70)
    print("🔧 REORGANIZADOR DE DATASET - CloudClassify13")
    print("=" * 70)
    
    if len(sys.argv) < 2:
        print("""
Uso:
    python reorganize_dataset.py analyze     - Ver sugerencias sin cambiar nada
    python reorganize_dataset.py apply       - Aplicar cambios sugeridos
    python reorganize_dataset.py revert      - Revertir última sesión
    python reorganize_dataset.py history     - Ver historial de cambios
    
Opciones:
    --min-conf=70     - Cambiar umbral mínimo de confianza (default: 60%)
    --ct-threshold=80 - Cambiar umbral para eliminar Ct (default: 75%)
""")
        return
    
    action = sys.argv[1].lower()
    
    # Parsear opciones
    min_conf = 60
    ct_threshold = 75
    
    for arg in sys.argv[2:]:
        if arg.startswith('--min-conf='):
            min_conf = int(arg.split('=')[1])
        elif arg.startswith('--ct-threshold='):
            ct_threshold = int(arg.split('=')[1])
    
    reorganizer = DatasetReorganizer()
    
    if action == 'analyze':
        suggestions = reorganizer.analyze_and_suggest(min_conf, ct_threshold)
        reorganizer.print_suggestions(suggestions)
        print("\n💡 Para aplicar estos cambios, ejecuta:")
        print(f"   python reorganize_dataset.py apply --min-conf={min_conf} --ct-threshold={ct_threshold}")
    
    elif action == 'apply':
        suggestions = reorganizer.analyze_and_suggest(min_conf, ct_threshold)
        reorganizer.print_suggestions(suggestions)
        
        if not suggestions['move'] and not suggestions['delete']:
            print("\n✅ No hay cambios para aplicar")
            return
        
        print("\n⚠️ ¿Deseas aplicar estos cambios? (s/n): ", end='')
        response = input().strip().lower()
        
        if response == 's':
            reorganizer.apply_changes(suggestions, dry_run=False)
        else:
            print("❌ Operación cancelada")
    
    elif action == 'revert':
        reorganizer.revert_last_session()
    
    elif action == 'history':
        reorganizer.show_history()
    
    else:
        print(f"❌ Acción no reconocida: {action}")


if __name__ == "__main__":
    main()
