from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
import joblib
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import os
import json
import shap
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import google.generativeai as genai
from django.http import HttpResponse
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# Load models once when the server starts
MODEL_DIR = os.path.join(settings.BASE_DIR.parent, 'ml_models')

# Load crop recommendation model
try:
    crop_model = joblib.load(os.path.join(MODEL_DIR, 'train_model.py'))
    print("✅ Crop recommendation model loaded successfully")
except Exception as e:
    crop_model = None
    print(f"❌ Failed to load crop model: {e}")

# Load CNN models
try:
    plant_disease_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'train_cnn_model.py'))
    print("✅ Plant disease model loaded successfully")
except Exception as e:
    plant_disease_model = None
    print(f"❌ Failed to load plant disease model: {e}")

try:
    pest_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'pest_model.h5'))
    print("✅ Pest detection model loaded successfully")
except Exception as e:
    pest_model = None
    print(f"❌ Failed to load pest model: {e}")

# Define class names for the CNN models based on PlantVillage dataset
PLANT_DISEASE_CLASSES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Define pest classes - expanded list for better coverage
PEST_CLASSES = [
    'Aphids', 'Army_worm', 'Beetle', 'Bollworm', 'Earthworm', 'Grasshopper',
    'Mites', 'Mosquito', 'Sawfly', 'Stem_borer', 'Caterpillar', 'Thrips',
    'Whitefly', 'Spider_mites', 'Cutworm', 'Leafhopper', 'Scale_insects',
    'Termites', 'Fruit_fly', 'Nematodes'
]

# Initialize Gemini AI
try:
    genai.configure(api_key=os.getenv('GEMINI_API_KEY', 'AIzaSyDz3M8RMJctNkp8CD5sdtuv_nvmmcZen1k'))
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    print(f"❌ Failed to initialize Gemini AI: {e}")
    gemini_model = None

# Initialize SHAP explainer for crop model
shap_explainer = None
if crop_model is not None:
    try:
        # Load sample data for SHAP background
        sample_data = np.array([[50, 25, 25, 25, 60, 6.5, 200]])  # Sample background data
        shap_explainer = shap.Explainer(crop_model, sample_data)
        print("✅ SHAP explainer initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize SHAP explainer: {e}")

@api_view(['POST'])
def predict_crop(request):
    """
    Predict the best crop based on soil and weather conditions
    Expected input: N, P, K, temperature, humidity, ph, rainfall
    """
    try:
        data = request.data
        features = [
            float(data.get('N', 0)),
            float(data.get('P', 0)),
            float(data.get('K', 0)),
            float(data.get('temperature', 0)),
            float(data.get('humidity', 0)),
            float(data.get('ph', 0)),
            float(data.get('rainfall', 0))
        ]
        
        if crop_model is None:
            return Response({'error': 'Crop model not loaded'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        # Convert to DataFrame to avoid sklearn warning
        feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        features_df = pd.DataFrame([features], columns=feature_names)
        
        prediction = crop_model.predict(features_df)
        confidence = crop_model.predict_proba(features_df).max()
        
        # Generate SHAP explanation
        shap_explanation = None
        if shap_explainer is not None:
            try:
                # Use DataFrame for SHAP to maintain consistency
                shap_values = shap_explainer(features_df)
                
                # Create SHAP visualization
                plt.figure(figsize=(10, 6))
                display_feature_names = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH', 'Rainfall']
                
                # Handle different SHAP output formats
                if hasattr(shap_values, 'values'):
                    shap_vals = shap_values.values[0]
                else:
                    shap_vals = shap_values[0] if isinstance(shap_values, list) else shap_values
                
                # Ensure shap_vals is a 1D array and matches feature count
                if hasattr(shap_vals, 'flatten'):
                    shap_vals = shap_vals.flatten()
                
                # Ensure we have exactly 7 values for 7 features
                if len(shap_vals) != len(display_feature_names):
                    shap_vals = shap_vals[:len(display_feature_names)]  # Take first 7 values
                
                # Sort features by absolute SHAP value for better visualization
                feature_importance = list(zip(display_feature_names, shap_vals))
                feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
                sorted_names, sorted_values = zip(*feature_importance)
                
                # Create bar plot with proper array handling
                colors = ['red' if float(x) < 0 else 'green' for x in sorted_values]
                plt.barh(range(len(sorted_names)), sorted_values, color=colors)
                plt.yticks(range(len(sorted_names)), sorted_names)
                plt.xlabel('SHAP Value (Impact on Prediction)')
                plt.title('Feature Importance for Crop Recommendation (Sorted by Impact)')
                plt.tight_layout()
                
                # Convert plot to base64
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
                buffer.seek(0)
                plot_data = buffer.getvalue()
                buffer.close()
                plt.close()
                
                shap_explanation = {
                    'plot': base64.b64encode(plot_data).decode(),
                    'values': {display_feature_names[i]: float(shap_vals[i]) for i in range(len(display_feature_names))},
                    'interpretation': generate_shap_interpretation(display_feature_names, shap_vals)
                }
            except Exception as e:
                print(f"SHAP explanation failed: {e}")
        
        return Response({
            'predicted_crop': prediction[0],
            'confidence': float(confidence),
            'input_features': {
                'N': features[0],
                'P': features[1],
                'K': features[2],
                'temperature': features[3],
                'humidity': features[4],
                'ph': features[5],
                'rainfall': features[6]
            },
            'shap_explanation': shap_explanation
        })
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
def predict_plant_disease(request):
    """
    Predict plant disease from uploaded image
    """
    try:
        if 'image' not in request.FILES:
            return Response({'error': 'No image provided'}, status=status.HTTP_400_BAD_REQUEST)
        
        if plant_disease_model is None:
            return Response({'error': 'Plant disease model not loaded'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        image_file = request.FILES['image']
        image = Image.open(image_file).convert('RGB')
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        predictions = plant_disease_model.predict(image_array)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        # Use actual class names from PlantVillage dataset
        if predicted_class < len(PLANT_DISEASE_CLASSES):
            predicted_disease = PLANT_DISEASE_CLASSES[predicted_class]
        else:
            predicted_disease = f"Unknown_Disease_{predicted_class}"
        
        # Generate AI solution for disease
        ai_solution = None
        if gemini_model:
            try:
                solution_prompt = f"""
                A plant has been diagnosed with {predicted_disease}.
                Provide practical treatment and prevention solutions for this plant disease.
                Include:
                1. Immediate treatment steps
                2. Prevention methods
                3. Organic/chemical treatment options
                4. Best practices for future prevention
                
                Keep the response concise and actionable for farmers.
                """
                solution_response = gemini_model.generate_content(solution_prompt)
                ai_solution = solution_response.text
            except Exception as e:
                print(f"AI solution generation failed: {e}")

        return Response({
            'predicted_disease': predicted_disease,
            'confidence': confidence,
            'ai_solution': ai_solution,
            'all_predictions': {
                PLANT_DISEASE_CLASSES[i] if i < len(PLANT_DISEASE_CLASSES) else f"Unknown_{i}": float(predictions[0][i]) 
                for i in range(len(predictions[0]))
            }
        })
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
def predict_pest(request):
    """
    Predict pest from uploaded image
    """
    try:
        if 'image' not in request.FILES:
            return Response({'error': 'No image provided'}, status=status.HTTP_400_BAD_REQUEST)
        
        if pest_model is None:
            return Response({'error': 'Pest model not loaded'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        image_file = request.FILES['image']
        image = Image.open(image_file).convert('RGB')
        
        # Enhanced image preprocessing for better pest detection
        image = image.resize((224, 224))
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        # Add slight contrast enhancement for better pest visibility
        image_array = np.clip(image_array * 1.1, 0, 1)
        image_array = np.expand_dims(image_array, axis=0)
        
        # Get predictions with error handling
        predictions = pest_model.predict(image_array, verbose=0)
        
        # Ensure predictions is in correct format
        if len(predictions.shape) > 1:
            pred_probs = predictions[0]
        else:
            pred_probs = predictions
            
        predicted_class = np.argmax(pred_probs)
        confidence = float(pred_probs[predicted_class])
        
        # Enhanced class mapping with bounds checking
        if predicted_class < len(PEST_CLASSES):
            predicted_pest = PEST_CLASSES[predicted_class]
        else:
            # If prediction exceeds known classes, map to closest or most likely
            predicted_class = predicted_class % len(PEST_CLASSES)
            predicted_pest = f"{PEST_CLASSES[predicted_class]}_variant"
            
        # Log prediction details for debugging
        print(f"Pest prediction: class={predicted_class}, confidence={confidence:.3f}, pest={predicted_pest}")
        print(f"Top 3 predictions: {sorted(enumerate(pred_probs), key=lambda x: x[1], reverse=True)[:3]}")
        
        # Generate AI solution for pest
        ai_solution = None
        if gemini_model:
            try:
                solution_prompt = f"""
                A pest has been identified as {predicted_pest}.
                Provide practical treatment and prevention solutions for this agricultural pest.
                Include:
                1. Immediate control measures
                2. Prevention strategies
                3. Organic/chemical treatment options
                4. Integrated pest management approaches
                5. Best practices for future prevention
                
                Keep the response concise and actionable for farmers.
                """
                solution_response = gemini_model.generate_content(solution_prompt)
                ai_solution = solution_response.text
            except Exception as e:
                print(f"AI solution generation failed: {e}")

        return Response({
            'predicted_pest': predicted_pest,
            'confidence': confidence,
            'ai_solution': ai_solution,
            'all_predictions': {
                PEST_CLASSES[i] if i < len(PEST_CLASSES) else f"Unknown_{i}": float(predictions[0][i]) 
                for i in range(len(predictions[0]))
            }
        })
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET'])
def health_check(request):
    """
    Health check endpoint
    """
    models_status = {
        'crop_model': crop_model is not None,
        'plant_disease_model': plant_disease_model is not None,
        'pest_model': pest_model is not None
    }
    
    return Response({
        'status': 'healthy',
        'models': models_status
    })

def generate_shap_interpretation(feature_names, shap_values):
    """Generate human-readable interpretation of SHAP values"""
    interpretations = []
    for i, (name, value) in enumerate(zip(feature_names, shap_values)):
        if abs(value) > 0.01:  # Only include significant features
            impact = "positively" if value > 0 else "negatively"
            interpretations.append(f"{name} impacts the prediction {impact} (value: {value:.3f})")
    return interpretations

@api_view(['POST'])
def calculate_profit(request):
    """Calculate profit based on area, cost, yield, and market price"""
    try:
        data = request.data
        area = float(data.get('area', 0))  # in acres
        cost_per_acre = float(data.get('cost_per_acre', 0))
        yield_per_acre = float(data.get('yield_per_acre', 0))  # in tons
        market_price = float(data.get('market_price', 0))  # per ton
        
        total_cost = area * cost_per_acre
        total_yield = area * yield_per_acre
        total_revenue = total_yield * market_price
        profit = total_revenue - total_cost
        profit_margin = (profit / total_revenue * 100) if total_revenue > 0 else 0
        
        return Response({
            'area': area,
            'total_cost': total_cost,
            'total_yield': total_yield,
            'total_revenue': total_revenue,
            'profit': profit,
            'profit_margin': profit_margin,
            'cost_per_acre': cost_per_acre,
            'yield_per_acre': yield_per_acre,
            'market_price': market_price
        })
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
def generate_cultivation_roadmap(request):
    """Generate cultivation roadmap using Gemini AI"""
    try:
        data = request.data
        crop = data.get('crop', '')
        location = data.get('location', 'general')
        season = data.get('season', 'current')
        
        # Validate required fields
        if not crop:
            return Response({'error': 'Crop name is required'}, status=status.HTTP_400_BAD_REQUEST)
        
        if not gemini_model:
            return Response({'error': 'Gemini AI not available'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        prompt = f"""
        Create a detailed cultivation roadmap for {crop} crop in {location} during {season} season.
        Provide a structured timeline with the following sections:
        1. Pre-planting preparation (soil preparation, seed selection)
        2. Planting phase (timing, spacing, depth)
        3. Growth phases (irrigation, fertilization, pest management)
        4. Harvesting (timing, methods, post-harvest handling)
        5. Best practices and tips
        
        IMPORTANT: Do not use any markdown formatting symbols like * (asterisks) or # (hash symbols) in your response.
        Use plain text with clear indentation and numbering only.
        Provide the response as clear, structured text without any special formatting characters.
        """
        
        response = gemini_model.generate_content(prompt)
        roadmap = response.text
        
        # Remove any markdown formatting symbols that might have been included
        roadmap = roadmap.replace('*', '').replace('#', '')
        # Clean up any double spaces that might result from removing symbols
        import re
        roadmap = re.sub(r'\s+', ' ', roadmap)
        roadmap = re.sub(r'\n\s*\n', '\n\n', roadmap)  # Preserve paragraph breaks
        
        return Response({
            'crop': crop,
            'location': location,
            'season': season,
            'roadmap': roadmap
        })
    except Exception as e:
        print(f"Roadmap generation error: {str(e)}")
        return Response({'error': f'Roadmap generation failed: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
def chat_with_ai(request):
    """Chat with Gemini AI for farming advice"""
    try:
        data = request.data
        message = data.get('message', '')
        
        if not gemini_model:
            return Response({'error': 'Gemini AI not available'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        # Add context for farming-specific responses
        farming_context = """
        You are KrishiAI, an expert agricultural assistant. Provide helpful, accurate, and practical advice for farmers.
        Focus on sustainable farming practices, crop management, pest control, soil health, and modern agricultural techniques.
        Keep responses concise but informative.
        
        User question: 
        """
        
        full_prompt = farming_context + message
        response = gemini_model.generate_content(full_prompt)
        
        return Response({
            'message': message,
            'response': response.text
        })
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
def generate_prediction_pdf(request):
    """Generate and download PDF report of crop prediction"""
    try:
        data = request.data
        prediction_data = data.get('prediction_data', {})
        
        response = HttpResponse(content_type='application/pdf')
        response['Content-Disposition'] = 'attachment; filename="crop_prediction_report.pdf"'
        
        # Create PDF
        p = canvas.Canvas(response, pagesize=letter)
        width, height = letter
        
        # Title
        p.setFont("Helvetica-Bold", 16)
        p.drawString(50, height - 50, "KrishiAI Crop Prediction Report")
        
        # Prediction details
        y_position = height - 100
        p.setFont("Helvetica", 12)
        
        if 'predicted_crop' in prediction_data:
            p.drawString(50, y_position, f"Recommended Crop: {prediction_data['predicted_crop']}")
            y_position -= 30
            
        if 'confidence' in prediction_data:
            p.drawString(50, y_position, f"Confidence: {prediction_data['confidence']:.2%}")
            y_position -= 30
            
        # Input features
        if 'input_features' in prediction_data:
            p.drawString(50, y_position, "Input Parameters:")
            y_position -= 20
            for key, value in prediction_data['input_features'].items():
                p.drawString(70, y_position, f"{key}: {value}")
                y_position -= 20
        
        p.showPage()
        p.save()
        
        return response
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
