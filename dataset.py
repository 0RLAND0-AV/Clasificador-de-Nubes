"""
CloudClassify13 - Dataset y DataLoaders
========================================
Carga de datos, preprocesamiento y data augmentation.

Archivo: dataset.py
Descripción: Define las transformaciones de imagen, carga el dataset
             e implementa los DataLoaders para entrenamiento y validación.
             Implementa técnicas de data augmentation como rotación,
             flip, brightness, contrast, etc.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from pathlib import Path
from config import (
    DATA_DIR, IMAGE_SIZE, BATCH_SIZE, NORMALIZE_MEAN, NORMALIZE_STD,
    USE_DATA_AUGMENTATION, AUGMENTATION_PARAMS, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT,
    CLASS_NAMES
)


class CloudDataset(Dataset):
    """
    Dataset personalizado para imágenes de nubes.
    
    Características:
    - Carga imágenes desde carpetas (1 carpeta = 1 clase)
    - Aplica transformaciones personalizadas
    - Soporta data augmentation
    - Normalización con estadísticas de ImageNet
    """
    
    def __init__(self, image_folder, transform=None, class_names=None):
        """
        Inicializa el dataset.
        
        Args:
            image_folder (str or Path): Ruta a la carpeta con imágenes
            transform (callable, optional): Transformaciones a aplicar
            class_names (list, optional): Nombres de las clases
        """
        self.image_folder = Path(image_folder)
        self.transform = transform
        self.class_names = class_names or sorted([d.name for d in self.image_folder.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
        
        # Recopilar todas las imágenes
        self.images = []
        self.labels = []
        
        for class_name in self.class_names:
            class_dir = self.image_folder / class_name
            if not class_dir.exists():
                continue
            
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.avif', '.tiff', '.tif']:
                    self.images.append(str(img_file))
                    self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        from PIL import Image
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Cargar imagen
        image = Image.open(img_path).convert('RGB')
        
        # Aplicar transformaciones
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(augment=True):
    """
    Define las transformaciones de imagen.
    
    Args:
        augment (bool): Si es True, aplica data augmentation
    
    Returns:
        dict: Diccionario con transformaciones para train y validation
    """
    
    # Transformaciones comunes
    common_transforms = [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    ]
    
    # Transformaciones de entrenamiento (con augmentation)
    train_transforms_list = [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    ]
    
    if augment and USE_DATA_AUGMENTATION:
        # Data Augmentation - técnicas para mejorar generalización
        if AUGMENTATION_PARAMS.get('random_horizontal_flip'):
            train_transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))
        
        if AUGMENTATION_PARAMS.get('random_vertical_flip'):
            train_transforms_list.append(transforms.RandomVerticalFlip(p=0.5))
        
        if AUGMENTATION_PARAMS.get('random_rotation'):
            degrees = AUGMENTATION_PARAMS['random_rotation']
            train_transforms_list.append(transforms.RandomRotation(degrees=degrees))
        
        if AUGMENTATION_PARAMS.get('random_crop'):
            train_transforms_list.append(transforms.RandomCrop(IMAGE_SIZE, padding=10))
        
        # ColorJitter: modifica brightness, contrast, saturation, hue
        if any([AUGMENTATION_PARAMS.get(f'random_{x}') for x in ['brightness', 'contrast', 'saturation', 'hue']]):
            train_transforms_list.append(transforms.ColorJitter(
                brightness=AUGMENTATION_PARAMS.get('random_brightness', 0),
                contrast=AUGMENTATION_PARAMS.get('random_contrast', 0),
                saturation=AUGMENTATION_PARAMS.get('random_saturation', 0),
                hue=AUGMENTATION_PARAMS.get('random_hue', 0)
            ))
        
        # GaussianBlur (opcional)
        if AUGMENTATION_PARAMS.get('gaussian_blur'):
            kernel_size = AUGMENTATION_PARAMS.get('gaussian_blur_kernel', 5)
            train_transforms_list.append(transforms.GaussianBlur(kernel_size=kernel_size))
    
    # Agregar normalización común
    train_transforms_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    ])
    
    # Transformaciones de validación/test (sin augmentation)
    val_transforms = transforms.Compose(common_transforms)
    train_transforms = transforms.Compose(train_transforms_list)
    
    return {
        'train': train_transforms,
        'val': val_transforms,
        'test': val_transforms
    }


def create_data_loaders(data_dir=DATA_DIR, batch_size=BATCH_SIZE, num_workers=4):
    """
    Crea DataLoaders para entrenamiento, validación y prueba.
    
    Args:
        data_dir (str or Path): Directorio con los datos
        batch_size (int): Tamaño del batch
        num_workers (int): Número de workers para carga de datos
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, class_names)
    """
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        raise ValueError(f"El directorio de datos no existe: {data_dir}")
    
    # Detectar si hay GPU disponible para pin_memory
    import torch
    use_pin_memory = torch.cuda.is_available()
    
    # Obtener transformaciones
    transforms_dict = get_transforms(augment=True)
    
    # Cargar dataset completo
    full_dataset = CloudDataset(
        image_folder=data_dir,
        transform=transforms_dict['train'],
        class_names=CLASS_NAMES
    )
    
    print(f"\n✓ Dataset cargado: {len(full_dataset)} imágenes")
    print(f"✓ Clases: {len(full_dataset.class_names)}")
    
    # Calcular tamaños
    total_size = len(full_dataset)
    train_size = int(total_size * TRAIN_SPLIT)
    val_size = int(total_size * VAL_SPLIT)
    test_size = total_size - train_size - val_size
    
    print(f"✓ Split: Train={train_size}, Val={val_size}, Test={test_size}")
    
    # Dividir dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Cambiar transformaciones para val y test (sin augmentation)
    val_dataset_no_aug = CloudDataset(
        image_folder=data_dir,
        transform=transforms_dict['val'],
        class_names=CLASS_NAMES
    )
    
    # Aplicar mismo split
    _, val_dataset, test_dataset = random_split(
        val_dataset_no_aug, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Crear DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    
    return train_loader, val_loader, test_loader, full_dataset.class_names


def get_single_image_loader(image_path, image_size=IMAGE_SIZE):
    """
    Crea un DataLoader para una single imagen (para inferencia).
    
    Args:
        image_path (str): Ruta a la imagen
        image_size (int): Tamaño de la imagen
    
    Returns:
        torch.Tensor: Tensor de la imagen procesada
    """
    from PIL import Image
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Agregar batch dimension
    
    return image_tensor


if __name__ == "__main__":
    # Test del dataset
    print("=" * 60)
    print("TEST DEL DATASET")
    print("=" * 60)
    
    try:
        train_loader, val_loader, test_loader, class_names = create_data_loaders()
        print(f"\n✓ DataLoaders creados exitosamente")
        print(f"✓ Train loader: {len(train_loader)} batches de {BATCH_SIZE}")
        print(f"✓ Val loader: {len(val_loader)} batches de {BATCH_SIZE}")
        print(f"✓ Test loader: {len(test_loader)} batches de {BATCH_SIZE}")
        print(f"✓ Clases: {class_names}")
        
        # Mostrar un batch
        images, labels = next(iter(train_loader))
        print(f"\nPrimer batch:")
        print(f"  - Imágenes shape: {images.shape}")
        print(f"  - Labels: {labels}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print(f"\nAsegúrate de haber colocado imágenes en la carpeta: {DATA_DIR}")
    
    print("\n" + "=" * 60)
