"""
Skin Disease Detection - Flask Backend API
============================================
Provides REST API endpoints for image classification.

Endpoints:
- POST /predict  : Upload image and get prediction
- GET  /health   : Health check
- GET  /classes  : Get available disease classes
"""

import os
import io
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from werkzeug.utils import secure_filename

# ==================== CONFIGURATION ====================
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'skin_disease_model.h5')
CLASS_INDICES_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'class_indices.npy')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
IMG_SIZE = (224, 224)

# Disease descriptions for frontend display
DISEASE_INFO = {
    'acne': {
        'description': 'A common skin condition that occurs when hair follicles become clogged with oil and dead skin cells. It causes pimples, blackheads, and whiteheads.',
        'symptoms': 'Whiteheads, blackheads, pimples, cysts, nodules.',
        'recommendation': 'Consider over-the-counter treatments with benzoyl peroxide or salicylic acid. Consult a dermatologist for persistent cases.'
    },
    'eczema': {
        'description': 'A condition that makes your skin red and itchy. It\'s common in children but can occur at any age. Eczema is long-lasting and tends to flare periodically.',
        'symptoms': 'Dry skin, itching, red to brownish-gray patches, small raised bumps.',
        'recommendation': 'Moisturize regularly, avoid triggers, and consult a dermatologist for prescription treatments.'
    },
    'melanoma': {
        'description': 'The most serious type of skin cancer, developing in the cells that produce melanin. Early detection is crucial for successful treatment.',
        'symptoms': 'Asymmetrical moles, irregular borders, color changes, diameter larger than 6mm, evolving appearance.',
        'recommendation': '⚠️ This may be serious. Please consult a dermatologist immediately for professional examination.'
    },
    'psoriasis': {
        'description': 'A skin disease that causes red, itchy scaly patches, most commonly on the knees, elbows, trunk, and scalp. It is a chronic disease with no cure.',
        'symptoms': 'Red patches of skin, silvery scales, dry cracked skin, itching, burning.',
        'recommendation': 'Consult a dermatologist for topical treatments, phototherapy, or systemic medications.'
    },
    'normal': {
        'description': 'The skin appears healthy with no significant abnormalities detected.',
        'symptoms': 'None - skin appears healthy.',
        'recommendation': 'Continue maintaining good skin hygiene and sun protection practices.'
    }
}

# ==================== MODEL LOADING ====================
model = None
class_names = []

def load_model():
    """Load the trained model and class indices."""
    global model, class_names
    
    try:
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"✅ Model loaded successfully from {MODEL_PATH}")
        else:
            print(f"⚠️  Model file not found at {MODEL_PATH}")
            print("Please train the model first using: python model/train_model.py")
            return False
        
        # Load class indices
        if os.path.exists(CLASS_INDICES_PATH):
            class_indices = np.load(CLASS_INDICES_PATH, allow_pickle=True).item()
            # Invert dictionary: index -> class name
            class_names = [k for k, v in sorted(class_indices.items(), key=lambda x: x[1])]
            print(f"✅ Classes loaded: {class_names}")
        else:
            # Fallback default classes
            class_names = ['acne', 'eczema', 'melanoma', 'psoriasis', 'normal']
            print(f"⚠️  Class indices not found. Using defaults: {class_names}")
        
        return True
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_bytes):
    """Preprocess image for model prediction."""
    try:
        # Open image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size
        image = image.resize(IMG_SIZE)
        
        # Convert to array and normalize
        img_array = np.array(image) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {str(e)}")

# ==================== API ROUTES ====================
@app.route('/')
def index():
    """Root endpoint with API info."""
    return jsonify({
        'message': 'Skin Disease Detection API',
        'version': '1.0.0',
        'endpoints': {
            'POST /predict': 'Upload image for prediction',
            'GET /health': 'Health check',
            'GET /classes': 'Get available disease classes'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    model_loaded = model is not None
    return jsonify({
        'status': 'healthy' if model_loaded else 'model_not_loaded',
        'model_loaded': model_loaded,
        'available_classes': class_names
    })

@app.route('/classes', methods=['GET'])
def get_classes():
    """Get list of available disease classes and their info."""
    classes_info = {}
    for cls in class_names:
        classes_info[cls] = DISEASE_INFO.get(cls, {
            'description': 'No description available.',
            'symptoms': 'N/A',
            'recommendation': 'Please consult a healthcare professional.'
        })
    
    return jsonify({
        'classes': class_names,
        'info': classes_info
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict skin disease from uploaded image."""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please train the model first.'
            }), 503
        
        # Check if image was uploaded
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image provided. Please upload an image file.'
            }), 400
        
        file = request.files['image']
        
        # Check if file was selected
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected.'
            }), 400
        
        # Validate file type
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Read and preprocess image
        image_bytes = file.read()
        processed_image = preprocess_image(image_bytes)
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx]) * 100
        
        # Get class name
        predicted_class = class_names[predicted_class_idx] if predicted_class_idx < len(class_names) else 'unknown'
        
        # Get all class probabilities
        all_probabilities = {
            class_names[i]: float(predictions[0][i]) * 100 
            for i in range(len(class_names))
        }
        
        # Sort by probability
        sorted_probs = dict(sorted(
            all_probabilities.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        
        # Get disease info
        disease_info = DISEASE_INFO.get(predicted_class, {
            'description': 'No description available.',
            'symptoms': 'N/A',
            'recommendation': 'Please consult a healthcare professional.'
        })
        
        return jsonify({
            'success': True,
            'prediction': predicted_class,
            'confidence': round(confidence, 2),
            'all_probabilities': sorted_probs,
            'info': disease_info
        })
    
    except ValueError as ve:
        return jsonify({
            'success': False,
            'error': str(ve)
        }), 400
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500

# ==================== MAIN ====================
if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Run Flask app
    print("\n" + "=" * 60)
    print("SKIN DISEASE DETECTION API SERVER")
    print("=" * 60)
    print("Starting server on http://localhost:5000")
    print("Press CTRL+C to stop")
    print("=" * 60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
