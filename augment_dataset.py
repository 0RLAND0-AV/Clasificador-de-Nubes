"""
CloudClassify13 - Generador de Dataset Aumentado
=================================================
Genera múltiples versiones de cada imagen usando Data Augmentation
para aumentar el tamaño del dataset de entrenamiento.

Uso:
    python augment_dataset.py --multiplier 10
"""

import argparse
from pathlib import Path
from PIL import Image
import random
from tqdm import tqdm

from torchvision import transforms
from config import DATA_DIR, IMAGE_SIZE, CLASS_NAMES


def get_augmentation_transforms():
    """
    Transformaciones CONSERVADORAS para data augmentation.
    Prioriza preservar características de las nubes.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        # Solo transformaciones geométricas suaves
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),  # Reducido de 30
        # Sin cambios de color agresivos
        transforms.ColorJitter(
            brightness=0.15,   # Muy reducido
            contrast=0.15,     # Muy reducido
            saturation=0.0,    # Desactivado
            hue=0.0           # Desactivado
        ),
        # Convertir a PIL para guardar
        transforms.ToPILImage() if False else lambda x: x,  # No-op, ya es PIL
    ])


def augment_dataset(multiplier=10, output_suffix='_augmented'):
    """
    Genera versiones aumentadas del dataset.
    
    Args:
        multiplier: Cuántas versiones generar por imagen original
        output_suffix: Sufijo para carpetas de salida
    """
    data_dir = Path(DATA_DIR)
    augment_transform = get_augmentation_transforms()
    
    print("=" * 70)
    print("GENERADOR DE DATASET AUMENTADO")
    print("=" * 70)
    print(f"Directorio base: {data_dir}")
    print(f"Multiplicador: {multiplier}x por imagen")
    print(f"Clases: {len(CLASS_NAMES)}")
    print("=" * 70 + "\n")
    
    total_generated = 0
    
    for class_name in CLASS_NAMES:
        class_dir = data_dir / class_name
        
        if not class_dir.exists():
            print(f"⚠ Clase {class_name} no encontrada, saltando...")
            continue
        
        # Obtener imágenes originales
        image_files = [
            f for f in class_dir.iterdir()
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
            and not f.stem.endswith('_aug')  # No aumentar imágenes ya aumentadas
        ]
        
        if not image_files:
            print(f"⚠ Clase {class_name} sin imágenes, saltando...")
            continue
        
        print(f"\n📂 Procesando clase: {class_name}")
        print(f"   Imágenes originales: {len(image_files)}")
        print(f"   Generando {len(image_files) * multiplier} imágenes...")
        
        class_generated = 0
        
        for img_file in tqdm(image_files, desc=f"  {class_name}"):
            try:
                # Cargar imagen original
                original_img = Image.open(img_file).convert('RGB')
                
                # Generar versiones aumentadas
                for i in range(multiplier):
                    # Aplicar transformaciones
                    augmented_img = augment_transform(original_img)
                    
                    # Guardar con nombre único
                    new_name = f"{img_file.stem}_aug{i:03d}{img_file.suffix}"
                    output_path = class_dir / new_name
                    
                    # Convertir tensor a PIL si es necesario
                    if hasattr(augmented_img, 'numpy'):
                        # Es tensor, no guardamos (esto no debería pasar con estas transforms)
                        pass
                    else:
                        augmented_img.save(output_path, quality=95)
                        class_generated += 1
                
            except Exception as e:
                print(f"\n   ✗ Error procesando {img_file.name}: {e}")
                continue
        
        total_generated += class_generated
        print(f"   ✓ Generadas: {class_generated} imágenes")
    
    print("\n" + "=" * 70)
    print(f"✓ COMPLETADO")
    print(f"Total de imágenes generadas: {total_generated}")
    print(f"Ahora puedes reentrenar el modelo con más datos.")
    print("=" * 70)


def clean_augmented_images():
    """
    Elimina todas las imágenes aumentadas (con _aug en el nombre).
    """
    data_dir = Path(DATA_DIR)
    
    print("\n⚠ LIMPIEZA DE IMÁGENES AUMENTADAS")
    print("=" * 70)
    
    total_deleted = 0
    
    for class_name in CLASS_NAMES:
        class_dir = data_dir / class_name
        
        if not class_dir.exists():
            continue
        
        aug_files = [
            f for f in class_dir.iterdir()
            if f.stem.endswith('_aug') or '_aug' in f.stem
        ]
        
        if aug_files:
            print(f"🗑️  {class_name}: {len(aug_files)} archivos")
            for f in aug_files:
                f.unlink()
                total_deleted += 1
    
    print(f"\n✓ Eliminadas {total_deleted} imágenes aumentadas")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera dataset aumentado")
    parser.add_argument(
        '--multiplier',
        type=int,
        default=10,
        help='Cuántas versiones aumentadas generar por imagen (default: 10)'
    )
    parser.add_argument(
        '--clean',
        action='store_true',
        help='Eliminar imágenes aumentadas existentes'
    )
    
    args = parser.parse_args()
    
    if args.clean:
        confirm = input("⚠ ¿Eliminar todas las imágenes aumentadas? (s/N): ")
        if confirm.lower() == 's':
            clean_augmented_images()
    else:
        augment_dataset(multiplier=args.multiplier)
