import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['SECRET_KEY'] = 'random-forest-placement-prediction-2025'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

MODEL_PATH = 'model_random_forest.pkl'
model_artifact = joblib.load(MODEL_PATH)
model = model_artifact['model']
feature_names = model_artifact['feature_names']
encoders = model_artifact['encoders']
y_encoder = model_artifact['y_encoder']
fill_values = model_artifact['fill_values']
metadata = model_artifact['metadata']

STATIC_IMAGES = [
    'Distribusi target.png',
    'Histogram Fitur Numerik.png',
    'Heatmap Korelasi.png',
    'Countplot Kelas Target.png',
    'Confusion Matrix.png',
    'Feature infortance.png',
    'ROC Curve AUC.png'
]

IMAGE_TITLES = {
    'Distribusi target.png': 'Distribusi Target Penempatan Kerja',
    'Histogram Fitur Numerik.png': 'Histogram Distribusi Fitur Numerik',
    'Heatmap Korelasi.png': 'Heatmap Korelasi Antar Fitur',
    'Countplot Kelas Target.png': 'Distribusi Fitur Kategorikal per Kelas',
    'Confusion Matrix.png': 'Confusion Matrix Model',
    'Feature infortance.png': 'Tingkat Kepentingan Fitur',
    'ROC Curve AUC.png': 'Kurva ROC dan Nilai AUC'
}

@app.route('/')
def index():
    return render_template('index.html', 
                         metadata=metadata,
                         feature_names=feature_names,
                         images=STATIC_IMAGES,
                         image_titles=IMAGE_TITLES)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        input_data = {}
        for feat in feature_names:
            value = data.get(feat)
            if value is None or value == '':
                if feat in fill_values:
                    input_data[feat] = fill_values[feat]
                else:
                    input_data[feat] = 0
            else:
                if feat in encoders:
                    classes = encoders[feat]['classes_']
                    if value in classes:
                        input_data[feat] = classes.index(value)
                    else:
                        input_data[feat] = 0
                else:
                    try:
                        input_data[feat] = float(value)
                    except:
                        input_data[feat] = 0
        
        input_df = pd.DataFrame([input_data])
        input_df = input_df[feature_names]
        
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        
        result_label = y_encoder['classes_'][prediction]
        confidence = float(probability[prediction]) * 100
        
        return jsonify({
            'success': True,
            'prediction': result_label,
            'confidence': round(confidence, 2),
            'probabilities': {
                y_encoder['classes_'][i]: round(float(probability[i]) * 100, 2)
                for i in range(len(probability))
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/model-info')
def model_info():
    feature_importance = model.feature_importances_
    fi_data = [
        {'feature': feature_names[i], 'importance': float(feature_importance[i])}
        for i in range(len(feature_names))
    ]
    fi_data.sort(key=lambda x: x['importance'], reverse=True)
    
    return jsonify({
        'metadata': metadata,
        'feature_importance': fi_data,
        'encoders': {k: v['classes_'] for k, v in encoders.items()},
        'target_classes': y_encoder['classes_']
    })

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': True})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
