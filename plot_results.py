#!/usr/bin/env python3
"""
CloudClassify13 - Generador de Gráficos de Resultados
Crea visualizaciones para el informe a partir del histórico de entrenamiento
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = {
    'train': '#2563eb',
    'val': '#10b981',
    'test': '#f59e0b'
}


def load_training_history(history_path='models/training_history.json'):
    """
    Cargar histórico de entrenamiento.
    
    Args:
        history_path: Ruta al archivo history JSON
        
    Returns:
        dict: Datos del histórico
    """
    if not Path(history_path).exists():
        print(f"⚠️  Archivo no encontrado: {history_path}")
        print("⚠️  Ejecuta primero: python main_train.py --mode train")
        return None
    
    with open(history_path, 'r') as f:
        return json.load(f)


def plot_loss_curves(history, output_dir='results'):
    """
    Graficar pérdida de entrenamiento y validación.
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    epochs = [h['epoch'] for h in history['epochs']]
    train_loss = [h['train_loss'] for h in history['epochs']]
    val_loss = [h['val_loss'] for h in history['epochs']]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_loss, marker='o', label='Training Loss', 
            color=COLORS['train'], linewidth=2)
    ax.plot(epochs, val_loss, marker='s', label='Validation Loss', 
            color=COLORS['val'], linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss (Cross Entropy)', fontsize=12, fontweight='bold')
    ax.set_title('Pérdida de Entrenamiento vs Validación', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Marcar mejor época
    best_epoch = history['best_epoch']
    best_val_loss = min(val_loss)
    ax.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5, label=f'Best Epoch: {best_epoch}')
    ax.plot(best_epoch, best_val_loss, 'r*', markersize=15)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_loss_curves.png', dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico guardado: {output_dir}/01_loss_curves.png")
    plt.close()


def plot_accuracy_curves(history, output_dir='results'):
    """
    Graficar accuracy de entrenamiento y validación.
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    epochs = [h['epoch'] for h in history['epochs']]
    train_acc = [h['train_acc'] for h in history['epochs']]
    val_acc = [h['val_acc'] for h in history['epochs']]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_acc, marker='o', label='Training Accuracy', 
            color=COLORS['train'], linewidth=2)
    ax.plot(epochs, val_acc, marker='s', label='Validation Accuracy', 
            color=COLORS['val'], linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Precisión de Entrenamiento vs Validación', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Marcar mejor época
    best_epoch = history['best_epoch']
    best_val_acc = max(val_acc)
    ax.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5, label=f'Best Epoch: {best_epoch}')
    ax.plot(best_epoch, best_val_acc, 'r*', markersize=15)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_accuracy_curves.png', dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico guardado: {output_dir}/02_accuracy_curves.png")
    plt.close()


def plot_convergence_analysis(history, output_dir='results'):
    """
    Análisis de convergencia (dos gráficos en uno).
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    epochs = [h['epoch'] for h in history['epochs']]
    train_loss = [h['train_loss'] for h in history['epochs']]
    val_loss = [h['val_loss'] for h in history['epochs']]
    train_acc = [h['train_acc'] for h in history['epochs']]
    val_acc = [h['val_acc'] for h in history['epochs']]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    ax1.plot(epochs, train_loss, marker='o', label='Training', 
            color=COLORS['train'], linewidth=2)
    ax1.plot(epochs, val_loss, marker='s', label='Validation', 
            color=COLORS['val'], linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax1.set_title('Convergencia de Pérdida', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(epochs, train_acc, marker='o', label='Training', 
            color=COLORS['train'], linewidth=2)
    ax2.plot(epochs, val_acc, marker='s', label='Validation', 
            color=COLORS['val'], linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax2.set_title('Convergencia de Precisión', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_convergence_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico guardado: {output_dir}/03_convergence_analysis.png")
    plt.close()


def plot_summary_statistics(history, output_dir='results'):
    """
    Estadísticas resumidas del entrenamiento.
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    best_epoch = history['best_epoch']
    best_val_acc = history['best_val_acc']
    final_train_acc = history['epochs'][-1]['train_acc']
    final_val_acc = history['epochs'][-1]['val_acc']
    final_test_acc = history.get('test_acc', 0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    # Crear tabla de texto
    stats_text = f"""
    ESTADÍSTICAS DE ENTRENAMIENTO
    ═════════════════════════════════════════════
    
    Mejor Época:                          {best_epoch}
    Mejor Accuracy (Validación):          {best_val_acc:.4f} ({best_val_acc*100:.2f}%)
    
    Accuracy Final (Entrenamiento):       {final_train_acc:.4f} ({final_train_acc*100:.2f}%)
    Accuracy Final (Validación):          {final_val_acc:.4f} ({final_val_acc*100:.2f}%)
    Accuracy (Test Set):                  {final_test_acc:.4f} ({final_test_acc*100:.2f}%)
    
    Total de Épocas:                      {len(history['epochs'])}
    
    Resumen:
    • El modelo convergió en época {best_epoch} de {len(history['epochs'])}
    • Mejor accuracy en validación: {best_val_acc*100:.2f}%
    • Diferencia train-val: {abs(final_train_acc - final_val_acc)*100:.2f}%
    • Estado: {'✓ Buen entrenamiento' if abs(final_train_acc - final_val_acc) < 0.15 else '⚠ Revisar overfitting'}
    ═════════════════════════════════════════════
    """
    
    ax.text(0.5, 0.5, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/04_summary_statistics.png', dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico guardado: {output_dir}/04_summary_statistics.png")
    plt.close()


def plot_epoch_comparison(history, output_dir='results'):
    """
    Comparación bar chart de primeras vs últimas épocas.
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    first_epoch = history['epochs'][0]
    last_epoch = history['epochs'][-1]
    best_epoch_data = history['epochs'][history['best_epoch']-1]
    
    epochs_labels = ['Epoch 1', f'Epoch {last_epoch["epoch"]}', f'Best (Epoch {history["best_epoch"]})']
    train_acc = [first_epoch['train_acc'], last_epoch['train_acc'], best_epoch_data['train_acc']]
    val_acc = [first_epoch['val_acc'], last_epoch['val_acc'], best_epoch_data['val_acc']]
    
    x = np.arange(len(epochs_labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, train_acc, width, label='Training', color=COLORS['train'], alpha=0.8)
    ax.bar(x + width/2, val_acc, width, label='Validation', color=COLORS['val'], alpha=0.8)
    
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Comparación de Accuracy: Inicio vs Final vs Mejor', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(epochs_labels)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Agregar valores en las barras
    for i, (ta, va) in enumerate(zip(train_acc, val_acc)):
        ax.text(i - width/2, ta + 0.01, f'{ta:.3f}', ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, va + 0.01, f'{va:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/05_epoch_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico guardado: {output_dir}/05_epoch_comparison.png")
    plt.close()


def generate_all_plots(history_path='models/training_history.json', output_dir='results'):
    """
    Generar todos los gráficos.
    """
    print("\n" + "="*60)
    print("GENERADOR DE GRÁFICOS DE RESULTADOS")
    print("="*60)
    
    history = load_training_history(history_path)
    if not history:
        return
    
    print(f"\n📊 Generando gráficos en {output_dir}/...")
    
    plot_loss_curves(history, output_dir)
    plot_accuracy_curves(history, output_dir)
    plot_convergence_analysis(history, output_dir)
    plot_summary_statistics(history, output_dir)
    plot_epoch_comparison(history, output_dir)
    
    print(f"\n✓ Todos los gráficos generados en: {output_dir}/")
    print(f"\n📁 Archivos creados:")
    print(f"   1. 01_loss_curves.png")
    print(f"   2. 02_accuracy_curves.png")
    print(f"   3. 03_convergence_analysis.png")
    print(f"   4. 04_summary_statistics.png")
    print(f"   5. 05_epoch_comparison.png")
    print(f"\n💡 Tip: Incluir estos gráficos en tu informe (sección Resultados)")


if __name__ == '__main__':
    generate_all_plots()
