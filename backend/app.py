import os
import logging
from datetime import datetime
from flask import Flask, request, jsonify
import pandas as pd
from joblib import load
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix
import numpy as np
from functools import wraps
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StrokePredictionAPI:
    def __init__(self):
        self.model_pipeline = None
        self.categorical_features = None
        self.numerical_features = None
        self.model_loaded = False
        self.load_model()
    
    def load_model(self):
        """Load the trained model with error handling."""
        try:
            model_path = os.getenv('MODEL_PATH', './stroke_prediction_model.joblib')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            
            model_data = load(model_path)
            self.model_pipeline = model_data['pipeline']
            self.categorical_features = model_data['categorical_features']
            self.numerical_features = model_data['numerical_features']
            self.model_loaded = True
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            self.model_loaded = False
            raise
    
    def validate_input(self, data):
        """Validate input data structure and types."""
        required_features = self.categorical_features + self.numerical_features
        
        # Check if all required features are present
        missing_features = [f for f in required_features if f not in data]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Validate data types and ranges
        validated_data = {}
        
        for feature in self.numerical_features:
            try:
                value = float(data[feature])
                
                # Add reasonable range checks
                if feature == 'age' and (value < 0 or value > 120):
                    raise ValueError(f"Age must be between 0 and 120, got {value}")
                elif feature == 'avg_glucose_level' and (value < 0 or value > 500):
                    raise ValueError(f"Glucose level must be between 0 and 500, got {value}")
                elif feature == 'bmi' and (value < 0 or value > 100):
                    raise ValueError(f"BMI must be between 0 and 100, got {value}")
                
                validated_data[feature] = value
                
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid value for {feature}: {data[feature]}")
        
        for feature in self.categorical_features:
            validated_data[feature] = data[feature]
        
        return validated_data
    
    def predict(self, data):
        """Make prediction with error handling."""
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        
        try:
            validated_data = self.validate_input(data)
            df = pd.DataFrame([validated_data])
            
            prediction = self.model_pipeline.predict(df)[0]
            prediction_proba = self.model_pipeline.predict_proba(df)[0]
            
            return {
                "prediction": int(prediction),
                "probability": {
                    "no_stroke": float(prediction_proba[0]),
                    "stroke": float(prediction_proba[1])
                },
                "confidence": float(max(prediction_proba))
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise

# Initialize the prediction service
try:
    prediction_service = StrokePredictionAPI()
except Exception as e:
    logger.error(f"Failed to initialize prediction service: {str(e)}")
    prediction_service = None

# Initialize Flask app
app = Flask(__name__)

# Load configuration from environment variables
app.config.update(
    PORT=int(os.getenv('PORT', 5001)),
    SERVER_IP=os.getenv('SERVER_IP', '127.0.0.1'),
    SECRET_KEY=os.getenv('SECRET_KEY', ''),  # Optional, not required for backend connection
    JSON_SORT_KEYS=False,
    JSONIFY_PRETTYPRINT_REGULAR=False
)

# Security middleware
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# CORS configuration for production
CORS(app, 
     origins=os.getenv('ALLOWED_ORIGINS', '*').split(','),
     methods=['GET', 'POST'],
     allow_headers=['Content-Type', 'Authorization'])

def rate_limit(max_requests=100, window=3600):
    """Simple rate limiting decorator."""
    requests = {}
    
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
            current_time = time.time()
            
            # Clean old requests
            requests[client_ip] = [req_time for req_time in requests.get(client_ip, []) 
                                 if current_time - req_time < window]
            
            # Check rate limit
            if len(requests.get(client_ip, [])) >= max_requests:
                return jsonify({"error": "Rate limit exceeded"}), 429
            
            # Add current request
            requests.setdefault(client_ip, []).append(current_time)
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

@app.before_request
def log_request_info():
    """Log incoming requests."""
    logger.info(f"Request: {request.method} {request.path} from {request.remote_addr}")

@app.after_request
def log_response_info(response):
    """Log response information."""
    logger.info(f"Response: {response.status_code}")
    return response

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    status = {
        "status": "healthy" if prediction_service and prediction_service.model_loaded else "unhealthy",
        "timestamp": datetime.utcnow().isoformat(),
        "model_loaded": prediction_service.model_loaded if prediction_service else False
    }
    return jsonify(status), 200 if status["status"] == "healthy" else 503

@app.route('/predict', methods=['POST'])
@rate_limit(max_requests=50, window=3600)  # 50 requests per hour per IP
def predict():
    """Main prediction endpoint."""
    if not prediction_service or not prediction_service.model_loaded:
        return jsonify({"error": "Service unavailable - model not loaded"}), 503
    
    try:
        # Validate request
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Make prediction
        result = prediction_service.predict(data)
        
        # Log successful prediction
        logger.info(f"Prediction made successfully: {result['prediction']}")
        
        return jsonify({
            "success": True,
            "data": result,
            "timestamp": datetime.utcnow().isoformat()
        }), 200
        
    except ValueError as e:
        logger.warning(f"Validation error: {str(e)}")
        return jsonify({"error": f"Validation error: {str(e)}"}), 400
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": "Prediction failed"}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Return model information."""
    if not prediction_service or not prediction_service.model_loaded:
        return jsonify({"error": "Service unavailable - model not loaded"}), 503
    
    return jsonify({
        "categorical_features": prediction_service.categorical_features,
        "numerical_features": prediction_service.numerical_features,
        "expected_columns": prediction_service.categorical_features + prediction_service.numerical_features,
        "model_status": "loaded"
    }), 200

@app.route('/', methods=['GET'])
def home():
    """Root endpoint."""
    return jsonify({
        "service": "Stroke Prediction API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "/health": "Health check",
            "/predict": "Make prediction (POST)",
            "/model-info": "Get model information"
        }
    })

if __name__ == "__main__":
    # Development server
    app.run(host=app.config['SERVER_IP'], port=app.config['PORT'], debug=False)
else:
    # Production WSGI server
    application = app