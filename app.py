"""
CloudClassify13 - Servidor Web Flask
=====================================
Interfaz web interactiva para clasificación de nubes.

Archivo: app.py
Descripción: Aplicación Flask que proporciona una interfaz web
             para subir imágenes y obtener predicciones en tiempo real.
"""

import os
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import torch
import traceback

# Agregar directorio padre al path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    WEB_PORT, MAX_FILE_SIZE, ALLOWED_EXTENSIONS, MODEL_PATH,
    CLOUD_CLASSES, PROJECT_NAME, PROJECT_VERSION, IMAGE_SIZE
)
from predict import CloudPredictor

# Inicializar Flask
app = Flask(__name__, template_folder='web', static_folder='web/static')
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
app.config['UPLOAD_FOLDER'] = Path(__file__).parent / 'uploads'

# Crear carpeta de uploads
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Cargar modelo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
try:
    predictor = CloudPredictor(model_path=MODEL_PATH, device=device)
    model_loaded = True
except Exception as e:
    print(f"⚠ Error cargando modelo: {e}")
    predictor = None
    model_loaded = False


def allowed_file(filename):
    """Verifica si el archivo tiene una extensión permitida."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Página principal."""
    return render_template('index.html',
                         project_name=PROJECT_NAME,
                         project_version=PROJECT_VERSION,
                         model_loaded=model_loaded)


@app.route('/api/predict', methods=['POST'])
def predict():
    """Endpoint de predicción."""
    
    if not model_loaded:
        return jsonify({'error': 'Modelo no cargado'}), 500
    
    # Verificar que hay un archivo
    if 'file' not in request.files:
        return jsonify({'error': 'No se proporcionó archivo'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Archivo vacío'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'Extensión no permitida. Use: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    try:
        # Guardar archivo temporalmente
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Realizar predicción
        result = predictor.predict_single(filepath, return_probabilities=True)
        
        # Preparar respuesta
        response = {
            'success': True,
            'predicted_class': result['predicted_class'],
            'predicted_class_name': result['predicted_class_name'],
            'confidence': result['confidence'],
            'confidence_percent': result['confidence_percent'],
            'top_predictions': result.get('top_predictions', []),
            'class_description': get_class_description(result['predicted_class'])
        }
        
        # Limpiar archivo
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Retorna información sobre las clases de nubes."""
    classes_info = []
    for code, name in CLOUD_CLASSES.items():
        classes_info.append({
            'code': code,
            'name': name,
            'description': get_class_description(code)
        })
    
    return jsonify({'classes': classes_info, 'total': len(classes_info)})


@app.route('/api/info', methods=['GET'])
def get_info():
    """Retorna información del proyecto."""
    return jsonify({
        'project_name': PROJECT_NAME,
        'version': PROJECT_VERSION,
        'model_loaded': model_loaded,
        'device': str(device),
        'num_classes': len(CLOUD_CLASSES),
        'image_size': IMAGE_SIZE
    })


def get_class_description(class_code):
    """Retorna descripción detallada de un tipo de nube."""
    descriptions = {
        'Ci': 'Cirros: Nubes altas, blancas y delicadas en forma de filamentos. Compuestas de cristales de hielo. Indican cambios climáticos en 24-48 horas.',
        'Cc': 'Cirrocúmulos: Nubes altas en forma de pequeñas masas blancas redondeadas. Típicamente en grupos. Clima variable.',
        'Cs': 'Cirrostratos: Velo transparente de hielo a gran altura. Produce halos alrededor del sol o la luna. Posible lluvia en 12-24 horas.',
        'Ac': 'Altocúmulos: Nubes medias en forma de capas o parches grises/blancos. Aspecto de olas o almenas.',
        'As': 'Altostratos: Capa grisácea o azulada a media altura. Permite ver el sol levemente. Lluvia moderada posible.',
        'Cu': 'Cúmulos: Nubes densas de buen tiempo con contornos bien definidos. Bases planas. Desarrollo vertical.',
        'Cb': 'Cumulonimbos: Nubes de tormenta muy densas. Pueden producir lluvia intensa, granizo y relámpagos.',
        'Ns': 'Nimboestratos: Capas oscuras de lluvia continua. Aspecto difuso por la precipitación. Lluvia o nieve segura.',
        'Sc': 'Estratocúmulos: Capas de nubes bajas grises y blancas. Pueden producir chubascos aislados.',
        'St': 'Estratos: Capas bajas uniformes grises. Pueden producir llovizna o garúa ligera.',
        'Ct': 'Contrails: Estelas de vapor de aviones. Se forman a gran altitud en condiciones húmedas.'
    }
    return descriptions.get(class_code, 'Descripción no disponible')


@app.errorhandler(413)
def request_entity_too_large(error):
    """Maneja archivos demasiado grandes."""
    return jsonify({'error': f'Archivo demasiado grande. Máximo: {MAX_FILE_SIZE / 1024 / 1024:.0f} MB'}), 413


@app.errorhandler(404)
def not_found(error):
    """Maneja rutas no encontradas."""
    return jsonify({'error': 'No encontrado'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Maneja errores internos."""
    return jsonify({'error': 'Error interno del servidor'}), 500


if __name__ == '__main__':
    print("=" * 60)
    print(f"{PROJECT_NAME} v{PROJECT_VERSION}")
    print("=" * 60)
    print(f"Iniciando servidor en: http://localhost:{WEB_PORT}")
    print(f"Dispositivo: {device}")
    print(f"Modelo cargado: {'✓' if model_loaded else '✗'}")
    print("=" * 60 + "\n")
    
    app.run(debug=True, port=WEB_PORT, host='0.0.0.0')
