#!/usr/bin/env python3
"""
Script de Validación del Proyecto CloudClassify13
Verifica que todas las dependencias e importaciones están correctas
"""

import sys
from pathlib import Path

def print_header(text):
    print("\n" + "="*60)
    print(text)
    print("="*60)

def test_imports():
    """Verifica que todas las importaciones funcionan."""
    print_header("1. VERIFICANDO IMPORTACIONES")
    
    errors = []
    
    # Test PyTorch
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"  - CUDA disponible: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  - CUDA version: {torch.version.cuda}")
    except ImportError as e:
        errors.append(f"✗ PyTorch: {e}")
        print(f"✗ PyTorch no encontrado")
    
    # Test TorchVision
    try:
        import torchvision
        print(f"✓ TorchVision {torchvision.__version__}")
    except ImportError as e:
        errors.append(f"✗ TorchVision: {e}")
        print(f"✗ TorchVision no encontrado")
    
    # Test Flask
    try:
        import flask
        print(f"✓ Flask {flask.__version__}")
    except ImportError as e:
        errors.append(f"✗ Flask: {e}")
        print(f"✗ Flask no encontrado")
    
    # Test Pillow
    try:
        import PIL
        print(f"✓ Pillow {PIL.__version__}")
    except ImportError as e:
        errors.append(f"✗ Pillow: {e}")
        print(f"✗ Pillow no encontrado")
    
    # Test NumPy
    try:
        import numpy
        print(f"✓ NumPy {numpy.__version__}")
    except ImportError as e:
        errors.append(f"✗ NumPy: {e}")
        print(f"✗ NumPy no encontrado")
    
    # Test Matplotlib (opcional pero recomendado)
    try:
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__}")
    except ImportError:
        print(f"⚠ Matplotlib no encontrado (opcional)")
    
    # Test TensorBoard (opcional pero recomendado)
    try:
        from torch.utils.tensorboard import SummaryWriter
        print(f"✓ TensorBoard disponible")
    except ImportError:
        print(f"⚠ TensorBoard no encontrado (opcional, instalar con: pip install tensorboard)")
    
    return errors

def test_project_modules():
    """Verifica que los módulos del proyecto se pueden importar."""
    print_header("2. VERIFICANDO MÓDULOS DEL PROYECTO")
    
    errors = []
    project_root = Path(__file__).parent
    
    modules = [
        'config',
        'model',
        'dataset',
        'train',
        'predict',
        'app',
        'main_train',
        'download_data'
    ]
    
    for module in modules:
        try:
            # Importar dinámicamente
            __import__(module)
            print(f"✓ {module}.py")
        except Exception as e:
            errors.append(f"✗ {module}.py: {e}")
            print(f"✗ {module}.py - Error: {str(e)[:50]}")
    
    return errors

def test_data_structure():
    """Verifica la estructura de directorios."""
    print_header("3. VERIFICANDO ESTRUCTURA DE DATOS")
    
    project_root = Path(__file__).parent
    data_dir = project_root / "data"
    
    cloud_classes = ['Ci', 'Cc', 'Cs', 'Ac', 'As', 'Cu', 'Cb', 'Ns', 'Sc', 'St', 'Ct']
    
    if not data_dir.exists():
        print(f"✗ Directorio data/ no existe")
        return ["Directorio data/ no existe"]
    
    print(f"✓ Directorio data/ existe")
    
    errors = []
    for cloud_class in cloud_classes:
        class_dir = data_dir / cloud_class
        if class_dir.exists():
            image_count = len(list(class_dir.glob("*.jpg"))) + len(list(class_dir.glob("*.png")))
            print(f"✓ {cloud_class}/ - {image_count} imágenes")
            if image_count == 0:
                errors.append(f"⚠ {cloud_class}/ está vacío")
        else:
            print(f"✗ {cloud_class}/ no existe")
            errors.append(f"{cloud_class}/ no existe")
    
    return errors

def test_model_directory():
    """Verifica el directorio de modelos."""
    print_header("4. VERIFICANDO DIRECTORIO DE MODELOS")
    
    project_root = Path(__file__).parent
    models_dir = project_root / "models"
    
    if not models_dir.exists():
        print(f"✓ Directorio models/ no existe (se creará al entrenar)")
        return []
    
    print(f"✓ Directorio models/ existe")
    
    # Buscar checkpoints
    checkpoints = list(models_dir.glob("*.pth")) + list(models_dir.glob("*.pt"))
    if checkpoints:
        print(f"  Checkpoints encontrados:")
        for cp in checkpoints:
            print(f"  - {cp.name}")
    else:
        print(f"  (Sin checkpoints guardados)")
    
    return []

def test_configuration():
    """Verifica la configuración."""
    print_header("5. VERIFICANDO CONFIGURACIÓN")
    
    try:
        from config import (
            NUM_CLASSES, CLASS_NAMES, EPOCHS, BATCH_SIZE, 
            LEARNING_RATE, OPTIMIZER, SCHEDULER
        )
        
        print(f"✓ Clases: {NUM_CLASSES}")
        print(f"✓ Épocas: {EPOCHS}")
        print(f"✓ Batch size: {BATCH_SIZE}")
        print(f"✓ Learning rate: {LEARNING_RATE}")
        print(f"✓ Optimizer: {OPTIMIZER}")
        print(f"✓ Scheduler: {SCHEDULER}")
        
        return []
    except Exception as e:
        print(f"✗ Error en configuración: {e}")
        return [f"Error en configuración: {e}"]

def main():
    """Ejecuta todas las validaciones."""
    print_header("VALIDACIÓN DEL PROYECTO CLOUDCLASSIFY13")
    
    all_errors = []
    
    # Ejecutar tests
    all_errors.extend(test_imports())
    all_errors.extend(test_project_modules())
    all_errors.extend(test_data_structure())
    all_errors.extend(test_model_directory())
    all_errors.extend(test_configuration())
    
    # Resumen final
    print_header("RESUMEN")
    
    if all_errors:
        print(f"❌ Se encontraron {len(all_errors)} problemas:\n")
        for i, error in enumerate(all_errors, 1):
            print(f"{i}. {error}")
        print("\n💡 Solución:")
        print("   pip install --upgrade -r requirements.txt")
        sys.exit(1)
    else:
        print("✅ Todas las validaciones pasaron correctamente!")
        print("\n🚀 El proyecto está listo para usar.")
        print("\nPróximos pasos:")
        print("1. python main_train.py --mode train --epochs 5 --verbose")
        print("2. python app.py")
        sys.exit(0)

if __name__ == "__main__":
    main()
