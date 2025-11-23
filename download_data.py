#!/usr/bin/env python3
"""
CloudClassify13 - Descargador de Imágenes de Entrenamiento
Descarga imágenes de URLs públicas para entrenar el modelo
"""

import os
import sys
import time
import urllib.request
from pathlib import Path
from urllib.error import URLError, HTTPError
import argparse

# URLs de ejemplo para descargar imágenes de nubes
# Puedes agregar más URLs según necesites
CLOUD_URLS = {
    'Ci': [
        'https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Cirrus_aviaticus_%286%29.jpg/1280px-Cirrus_aviaticus_%286%29.jpg',
        'https://upload.wikimedia.org/wikipedia/commons/thumb/1/16/Cirrus_fibratus.jpg/1024px-Cirrus_fibratus.jpg',
    ],
    'Cc': [
        'https://upload.wikimedia.org/wikipedia/commons/thumb/8/8f/Cirrocumulus_clouds.jpg/1024px-Cirrocumulus_clouds.jpg',
    ],
    'Cs': [
        'https://upload.wikimedia.org/wikipedia/commons/thumb/d/d6/Cirrostratus_fibratus.jpg/1024px-Cirrostratus_fibratus.jpg',
    ],
    'Ac': [
        'https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/Altocumulus_clouds.jpg/1024px-Altocumulus_clouds.jpg',
    ],
    'As': [
        'https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/Altostratus_clouds.jpg/1024px-Altostratus_clouds.jpg',
    ],
    'Cu': [
        'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3c/Cumulus_clouds_in_fair_weather.jpg/1024px-Cumulus_clouds_in_fair_weather.jpg',
    ],
    'Cb': [
        'https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/Cumulonimbus_clouds.jpg/1024px-Cumulonimbus_clouds.jpg',
    ],
    'Ns': [
        'https://upload.wikimedia.org/wikipedia/commons/thumb/f/f6/Nimbostratus_clouds.jpg/1024px-Nimbostratus_clouds.jpg',
    ],
    'Sc': [
        'https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/Stratocumulus_clouds.jpg/1024px-Stratocumulus_clouds.jpg',
    ],
    'St': [
        'https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/Stratus_clouds.jpg/1024px-Stratus_clouds.jpg',
    ],
    'Ct': [
        'https://upload.wikimedia.org/wikipedia/commons/thumb/e/e9/Contrails_from_aircraft.jpg/1024px-Contrails_from_aircraft.jpg',
    ],
}


def download_image(url, filepath, timeout=10):
    """
    Descargar imagen de URL.
    
    Args:
        url (str): URL de la imagen
        filepath (str): Ruta donde guardar la imagen
        timeout (int): Timeout en segundos
        
    Returns:
        bool: True si descarga exitosa, False en caso contrario
    """
    try:
        urllib.request.urlretrieve(url, filepath, timeout=timeout)
        return True
    except (URLError, HTTPError, Exception) as e:
        print(f"  ⚠️  Error descargando {url}: {e}")
        return False


def download_cloud_images(data_dir='data', max_per_class=5, verbose=False):
    """
    Descargar imágenes de ejemplo para cada clase de nube.
    
    Args:
        data_dir (str): Directorio donde guardar datos
        max_per_class (int): Máximo de imágenes por clase
        verbose (bool): Salida detallada
    """
    data_path = Path(data_dir)
    
    print("\n" + "="*60)
    print("DESCARGADOR DE IMÁGENES DE NUBES")
    print("="*60)
    
    total_downloaded = 0
    total_failed = 0
    
    for cloud_class, urls in CLOUD_URLS.items():
        class_dir = data_path / cloud_class
        class_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📥 Descargando {cloud_class} ({len(urls)} URLs)...")
        
        downloaded = 0
        for idx, url in enumerate(urls, 1):
            if downloaded >= max_per_class:
                print(f"  ℹ️  Límite de {max_per_class} imágenes alcanzado")
                break
            
            # Nombre del archivo
            filename = f"{cloud_class}_{idx:03d}.jpg"
            filepath = class_dir / filename
            
            if filepath.exists():
                print(f"  ✓ {filename} (ya existe)")
                downloaded += 1
                total_downloaded += 1
                continue
            
            # Descargar
            if verbose:
                print(f"  Descargando {filename}...")
            
            if download_image(url, str(filepath)):
                print(f"  ✓ {filename}")
                downloaded += 1
                total_downloaded += 1
            else:
                total_failed += 1
            
            time.sleep(0.5)  # Delay para no saturar servidor
        
        print(f"  Completado: {downloaded}/{len(urls)} imágenes")
    
    print(f"\n" + "="*60)
    print(f"✓ DESCARGA COMPLETADA")
    print("="*60)
    print(f"  • Imágenes descargadas: {total_downloaded}")
    print(f"  • Errores: {total_failed}")
    print(f"  • Ubicación: {data_dir}/")
    
    return total_downloaded, total_failed


def create_sample_structure(data_dir='data'):
    """
    Crear estructura de directorios para datos.
    
    Args:
        data_dir (str): Directorio raíz de datos
    """
    data_path = Path(data_dir)
    cloud_classes = ['Ci', 'Cc', 'Cs', 'Ac', 'As', 'Cu', 'Cb', 'Ns', 'Sc', 'St', 'Ct']
    
    print(f"\n📁 Creando estructura de directorios en {data_dir}...")
    
    for cloud_class in cloud_classes:
        class_dir = data_path / cloud_class
        class_dir.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {cloud_class}/")
    
    print(f"✓ Estructura creada")


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description='CloudClassify13 - Descargador de imágenes de nubes'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Directorio para guardar imágenes (default: data)'
    )
    parser.add_argument(
        '--max-per-class',
        type=int,
        default=5,
        help='Máximo de imágenes por clase (default: 5)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Salida detallada'
    )
    parser.add_argument(
        '--create-only',
        action='store_true',
        help='Solo crear estructura, no descargar'
    )
    
    args = parser.parse_args()
    
    # Crear estructura
    create_sample_structure(args.data_dir)
    
    # Descargar (si no es create-only)
    if not args.create_only:
        download_cloud_images(
            data_dir=args.data_dir,
            max_per_class=args.max_per_class,
            verbose=args.verbose
        )
    
    print(f"\n⚠️  NOTA:")
    print(f"  Este script descarga imágenes de ejemplo.")
    print(f"  Para mejores resultados, agrega más imágenes a {args.data_dir}/")
    print(f"  Estructura esperada:")
    print(f"    {args.data_dir}/Ci/*.jpg")
    print(f"    {args.data_dir}/Cc/*.jpg")
    print(f"    ... etc")


if __name__ == '__main__':
    main()
