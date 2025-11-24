#!/usr/bin/env python3
"""
CloudClassify13 - Script Principal de Entrenamiento
Grupo #13 - Proyecto de Clasificación de Nubes con CNN
"""

import os
import sys
import json
import argparse
from pathlib import Path
import torch

# Import custom modules
import config
from model import CloudCNN
from dataset import create_data_loaders
from train import CloudClassifierTrainer
from predict import CloudPredictor


def create_argument_parser():
    """Crear parser de argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description='CloudClassify13 - Clasificador de Nubes con CNN'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        choices=['train', 'evaluate', 'predict'],
        help='Modo de ejecución (entrenar, evaluar, o predecir)'
    )
    parser.add_argument(
        '--image',
        type=str,
        default=None,
        help='Ruta a imagen para predicción (solo con --mode predict)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Ruta a checkpoint de modelo guardado'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Número de épocas (sobrescribe config.EPOCHS)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Tamaño de batch (sobrescribe config.BATCH_SIZE)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Tasa de aprendizaje (sobrescribe config.LEARNING_RATE)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Dispositivo (auto = GPU si disponible, sino CPU)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Modo verbose (salida detallada)'
    )
    return parser


def setup_directories(config):
    """Crear directorios necesarios si no existen."""
    Path(config.MODELS_DIR).mkdir(parents=True, exist_ok=True)
    Path(config.DATA_DIR).mkdir(parents=True, exist_ok=True)
    
    # Crear carpetas para clases si no existen
    for class_name in config.CLOUD_CLASSES:
        class_dir = Path(config.DATA_DIR) / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Directorios configurados en: {config.PROJECT_ROOT}")


def train_model(config, args):
    """Entrenar el modelo."""
    print("\n" + "="*60)
    print("ENTRENAMIENTO DE MODELO CNN")
    print("="*60)
    
    # Actualizar configuración con argumentos
    if args.epochs:
        config.EPOCHS = args.epochs
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.lr:
        config.LEARNING_RATE = args.lr
    
    # Configurar dispositivo
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"\n📊 Configuración:")
    print(f"  • Dispositivo: {device}")
    print(f"  • Épocas: {config.EPOCHS}")
    print(f"  • Batch size: {config.BATCH_SIZE}")
    print(f"  • Learning rate: {config.LEARNING_RATE}")
    print(f"  • Clases: {len(config.CLOUD_CLASSES)}")
    
    # Crear modelo
    print(f"\n🏗️  Creando modelo...")
    model = CloudCNN(num_classes=len(config.CLOUD_CLASSES))
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  • Parámetros totales: {total_params:,}")
    print(f"  • Parámetros entrenable: {trainable_params:,}")
    
    # Cargar datos
    print(f"\n📦 Cargando datos...")
    try:
        train_loader, val_loader, test_loader, class_names = create_data_loaders(
            data_dir=config.DATA_DIR,
            batch_size=config.BATCH_SIZE,
            num_workers=0  # Windows compatible
        )
        print(f"  • Train samples: {len(train_loader.dataset)}")
        print(f"  • Val samples: {len(val_loader.dataset)}")
        print(f"  • Test samples: {len(test_loader.dataset)}")
    except Exception as e:
        print(f"❌ Error al cargar datos: {e}")
        print(f"⚠️  Asegúrate de tener imágenes en: {config.DATA_DIR}")
        return
    
    # Crear entrenador
    print(f"\n⚙️  Inicializando entrenador...")
    trainer = CloudClassifierTrainer(
        model=model,
        device=device,
        learning_rate=config.LEARNING_RATE,
        optimizer_name=config.OPTIMIZER,
        scheduler_name=config.SCHEDULER,
        checkpoint_dir=config.MODELS_DIR,
        verbose=args.verbose
    )
    
    # Entrenar
    print(f"\n🚀 Iniciando entrenamiento...\n")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.EPOCHS
    )
    
    # Guardar historia
    history_path = Path(config.MODELS_DIR) / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n✓ Historia guardada en: {history_path}")
    
    # Evaluar en test set
    print(f"\n📈 Evaluando en conjunto de prueba...")
    test_loss, test_accuracy = trainer.validate(test_loader)
    print(f"  • Loss en test: {test_loss:.4f}")
    print(f"  • Accuracy en test: {test_accuracy:.2f}%")
    
    # Resumen final
    print(f"\n" + "="*60)
    print(f"✓ ENTRENAMIENTO COMPLETADO")
    print(f"="*60)
    print(f"  • Mejor modelo guardado en: {trainer.best_checkpoint_path}")
    print(f"  • Historia de entrenamiento: {history_path}")


def evaluate_model(config, args):
    """Evaluar modelo en conjunto de prueba."""
    print("\n" + "="*60)
    print("EVALUACIÓN DE MODELO")
    print("="*60)
    
    if not args.checkpoint:
        print("❌ Error: especifica --checkpoint para evaluar")
        return
    
    # Configurar dispositivo
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"\n📊 Configuración:")
    print(f"  • Dispositivo: {device}")
    print(f"  • Checkpoint: {args.checkpoint}")
    
    # Crear modelo
    print(f"\n🏗️  Cargando modelo...")
    model = CloudCNN(num_classes=len(config.CLOUD_CLASSES))
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Cargar datos
    print(f"\n📦 Cargando datos...")
    _, _, test_loader, _ = create_data_loaders(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=0
    )
    
    # Evaluar
    trainer = CloudClassifierTrainer(
        model=model,
        device=device,
        checkpoint_dir=config.MODELS_DIR
    )
    
    test_loss, test_accuracy = trainer.validate(test_loader)
    print(f"\n✓ Loss en test set: {test_loss:.4f}")
    print(f"✓ Accuracy en test set: {test_accuracy:.2f}%")


def predict_image(config, args):
    """Realizar predicción en imagen."""
    print("\n" + "="*60)
    print("PREDICCIÓN")
    print("="*60)
    
    if not args.image:
        print("❌ Error: especifica --image para predecir")
        return
    
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"❌ Error: imagen no encontrada: {image_path}")
        return
    
    # Configurar dispositivo
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # Buscar checkpoint más reciente si no se especifica
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        models_dir = Path(config.MODELS_DIR)
        checkpoints = list(models_dir.glob('checkpoint_*.pt'))
        if not checkpoints:
            print(f"❌ Error: no se encontraron checkpoints en {config.MODELS_DIR}")
            return
        checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
    
    print(f"\n📊 Configuración:")
    print(f"  • Dispositivo: {device}")
    print(f"  • Imagen: {image_path}")
    print(f"  • Checkpoint: {checkpoint_path}")
    
    # Crear predictor
    print(f"\n🏗️  Cargando modelo...")
    predictor = CloudPredictor(
        checkpoint_path=str(checkpoint_path),
        num_classes=len(config.CLOUD_CLASSES),
        class_names=config.CLOUD_CLASSES,
        device=device
    )
    
    # Realizar predicción
    print(f"\n🔮 Prediciendo...")
    prediction, confidence = predictor.predict_single(str(image_path))
    
    print(f"\n✓ Predicción:")
    print(f"  • Clase: {prediction}")
    print(f"  • Confianza: {confidence:.2%}")
    
    # Top-K predicciones
    top_k = predictor.predict_batch([str(image_path)], k=3)
    if top_k:
        print(f"\n  Top 3 predicciones:")
        for i, pred in enumerate(top_k[0], 1):
            print(f"    {i}. {pred['class']}: {pred['confidence']:.2%}")


def main():
    """Función principal."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Setup
    setup_directories(config)
    
    # Ejecutar según modo
    if args.mode == 'train':
        train_model(config, args)
    elif args.mode == 'evaluate':
        evaluate_model(config, args)
    elif args.mode == 'predict':
        predict_image(config, args)


if __name__ == '__main__':
    main()
