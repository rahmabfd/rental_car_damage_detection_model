# app.py (Flask backend with MongoDB and Parallel Roboflow Models)

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pymongo import MongoClient
from bson import ObjectId
import os
import uuid
import json
from datetime import datetime
import requests
import cv2
import numpy as np
import base64
from PIL import Image, ImageDraw
import io
import gridfs
from roboflow import Roboflow
import concurrent.futures
import threading
from typing import List, Dict, Any

app = Flask(__name__)
CORS(app)

# Configuration for multiple Roboflow models
ROBOFLOW_API_KEY = '5tVmQJieqoJ9VsxVsCvc'
MODELS_CONFIG = [
    {
        'name': 'model_1',
        'workspace': 'da5-ezin-gkxmo',
        'project': 'final-project-jukje-3vx9q',  # Replace with your first model
        'version': 2,
        'confidence': 5,
        'overlap': 50
    },
    {
        'name': 'model_2', 
        'workspace': 'da5-ezin-gkxmo',
        'project': 'clean-ycy4j-nifjb',  # Replace with your second model
        'version': 2,
        'confidence': 10,
        'overlap': 40
    }
]

# Global variables for models
models = {}
models_lock = threading.Lock()

def initialize_roboflow_models():
    """Initialize multiple Roboflow models"""
    global models
    
    try:
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        
        for config in MODELS_CONFIG:
            try:
                project = rf.workspace(config['workspace']).project(config['project'])
                model = project.version(config['version']).model
                
                with models_lock:
                    models[config['name']] = {
                        'model': model,
                        'config': config
                    }
                
                print(f"✅ {config['name']} loaded successfully from {config['workspace']}/{config['project']}")
                
            except Exception as e:
                print(f"❌ Failed to load {config['name']}: {str(e)}")
                with models_lock:
                    models[config['name']] = None
                    
        print(f"✅ Loaded {len([m for m in models.values() if m is not None])} out of {len(MODELS_CONFIG)} models")
        
    except Exception as e:
        print(f"❌ Failed to initialize Roboflow: {str(e)}")
        models = {}

# Initialize models at startup
initialize_roboflow_models()

# MongoDB Configuration
MONGO_URI = 'mongodb://localhost:27017/'  # Update with your MongoDB URI
DB_NAME = 'car_rental_db'
COLLECTION_NAME = 'rentals'

app.config['UPLOAD_FOLDER'] = 'UPLOAD_FOLDER'

# Ensure upload directory exists
os.makedirs('UPLOAD_FOLDER', exist_ok=True)

# MongoDB setup
client = None
db = None
rentals_collection = None
fs = None

def init_mongodb():
    global client, db, rentals_collection, fs
    try:
        client = MongoClient(MONGO_URI)
        # Test the connection
        client.admin.command('ping')
        db = client[DB_NAME]
        rentals_collection = db[COLLECTION_NAME]
        fs = gridfs.GridFS(db)  # For storing large files if needed
        print("✅ Connected to MongoDB successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to connect to MongoDB: {str(e)}")
        client = None
        db = None
        rentals_collection = None
        fs = None
        return False

# Initialize MongoDB connection
mongo_connected = init_mongodb()

def run_single_model_inference(model_name: str, model_data: Dict, image_path: str) -> Dict[str, Any]:
    """Run inference on a single model"""
    try:
        if model_data is None:
            print(f"Model {model_name} is not loaded")
            return {'model_name': model_name, 'predictions': [], 'error': 'Model not loaded'}
        
        model = model_data['model']
        config = model_data['config']
        
        # Run inference with model-specific parameters
        predictions = model.predict(
            image_path, 
            confidence=config['confidence'], 
            overlap=config['overlap']
        ).json()
        
        return {
            'model_name': model_name,
            'predictions': predictions.get('predictions', []),
            'config': config,
            'error': None
        }
        
    except Exception as e:
        print(f"Error in {model_name} inference: {str(e)}")
        return {
            'model_name': model_name,
            'predictions': [],
            'error': str(e)
        }

def merge_model_predictions(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge predictions from multiple models and remove duplicates"""
    all_predictions = []
    
    # Collect all predictions from all models
    for result in results:
        if result['error'] is None:
            for prediction in result['predictions']:
                prediction['source_model'] = result['model_name']
                all_predictions.append(prediction)
    
    # Remove duplicate detections using Non-Maximum Suppression (NMS)
    if not all_predictions:
        return []
    
    # Group predictions by class
    predictions_by_class = {}
    for pred in all_predictions:
        class_name = pred['class']
        if class_name not in predictions_by_class:
            predictions_by_class[class_name] = []
        predictions_by_class[class_name].append(pred)
    
    # Apply NMS for each class
    final_predictions = []
    for class_name, class_predictions in predictions_by_class.items():
        # Sort by confidence (highest first)
        class_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Apply simple NMS
        kept_predictions = []
        for pred in class_predictions:
            bbox1 = [
                pred['x'] - pred['width'] / 2,
                pred['y'] - pred['height'] / 2,
                pred['x'] + pred['width'] / 2,
                pred['y'] + pred['height'] / 2
            ]
            
            # Check overlap with already kept predictions
            should_keep = True
            for kept_pred in kept_predictions:
                bbox2 = [
                    kept_pred['x'] - kept_pred['width'] / 2,
                    kept_pred['y'] - kept_pred['height'] / 2,
                    kept_pred['x'] + kept_pred['width'] / 2,
                    kept_pred['y'] + kept_pred['height'] / 2
                ]
                
                # If overlap is too high, don't keep this prediction
                if bbox_overlap(bbox1, bbox2) > 0.3:  # 30% overlap threshold
                    should_keep = False
                    break
            
            if should_keep:
                kept_predictions.append(pred)
        
        final_predictions.extend(kept_predictions)
    
    return final_predictions

def analyze_damage_with_parallel_models(image_path: str) -> Dict[str, Any]:
    """
    Analyze an image for car damage using multiple Roboflow models in parallel.
    Merges results to appear as a single model output.
    """
    if not models or all(m is None for m in models.values()):
        print("No models loaded, using fallback")
        return create_fallback_analysis(image_path)
    
    try:
        # Read and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return create_fallback_analysis(image_path)
        
        # Validate resolution
        h, w = image.shape[:2]
        if h < 640 or w < 640:
            print(f"Warning: Image resolution {w}x{h} is low. Recommended: ≥640x640")
        
        # Resize to 640x640, preserving aspect ratio with padding
        target_size = 640
        scale = min(target_size / w, target_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to square
        padded = np.full((target_size, target_size, 3), 128, dtype=np.uint8)  # Gray padding
        x_offset = (target_size - new_w) // 2
        y_offset = (target_size - new_h) // 2
        padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        # Save padded image temporarily for inference
        temp_path = os.path.join('UPLOAD_FOLDER', f'temp_padded_{uuid.uuid4().hex[:8]}.jpg')
        cv2.imwrite(temp_path, padded, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        
        # Run all models in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(models)) as executor:
            futures = []
            
            with models_lock:
                for model_name, model_data in models.items():
                    future = executor.submit(run_single_model_inference, model_name, model_data, temp_path)
                    futures.append(future)
            
            # Collect results
            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error getting result from future: {str(e)}")
        
        # Merge predictions from all models
        merged_predictions = merge_model_predictions(results)
        print(f"Merged predictions from {len(results)} models: {len(merged_predictions)} total detections")
        
        # Process merged predictions and scale back to original image coordinates
        has_damage = len(merged_predictions) > 0
        damages = []
        
        for prediction in merged_predictions:
            # Scale bounding boxes back to original image coordinates
            x_center = (prediction['x'] - x_offset) / scale
            y_center = (prediction['y'] - y_offset) / scale
            width = prediction['width'] / scale
            height = prediction['height'] / scale
            
            damages.append({
                'class': prediction['class'],
                'confidence': prediction['confidence'],
                'bbox': [
                    x_center - width / 2,   # x_min
                    y_center - height / 2,  # y_min
                    x_center + width / 2,   # x_max
                    y_center + height / 2   # y_max
                ],
                'source_model': prediction.get('source_model', 'unknown')
            })
        
        # Create annotated image using original image
        annotated_image_path = create_annotated_image(image_path, damages)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Return unified result (appears as single model)
        result = {
            'has_damage': has_damage,
            'damages': damages,
            'original_image': f'/uploads/{os.path.basename(image_path)}',
            'annotated_image': f'/uploads/{os.path.basename(annotated_image_path)}' if annotated_image_path else None,
            'prediction_count': len(damages),
            'models_used': [r['model_name'] for r in results if r['error'] is None],
            'note': f'Analysis completed using {len([r for r in results if r["error"] is None])} models' if has_damage else 'No damages detected'
        }
        
        print(f"Final analysis result: {result['prediction_count']} damages detected using {len(result['models_used'])} models")
        return result
        
    except Exception as e:
        print(f"Error in parallel damage analysis: {str(e)}")
        return create_fallback_analysis(image_path)

def create_fallback_analysis(image_path):
    """
    Create a fallback analysis when Roboflow API fails
    """
    return {
        'has_damage': False,
        'damages': [],
        'original_image': f'/uploads/{os.path.basename(image_path)}',
        'annotated_image': f'/uploads/{os.path.basename(image_path)}',  # Show original image as fallback
        'prediction_count': 0,
        'models_used': [],
        'note': 'Analysis completed with fallback method - no models available'
    }

def create_annotated_image(image_path, damages):
    """
    Create an annotated image with damage bounding boxes
    """
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Convert BGR to RGB for PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil_image)
        
        # Draw bounding boxes with different colors for different models
        colors = ['red', 'orange', 'yellow', 'pink', 'purple', 'blue', 'green', 'cyan']
        model_colors = {}
        color_idx = 0
        
        for i, damage in enumerate(damages):
            bbox = damage['bbox']
            
            # Assign color based on source model
            source_model = damage.get('source_model', 'unknown')
            if source_model not in model_colors:
                model_colors[source_model] = colors[color_idx % len(colors)]
                color_idx += 1
            
            color = model_colors[source_model]
            
            # Draw rectangle
            draw.rectangle(
                [bbox[0], bbox[1], bbox[2], bbox[3]], 
                outline=color, 
                width=3
            )
            
            # Draw label (without showing source model to maintain unified appearance)
            label = f"{damage['class']} ({damage['confidence']:.2f})"
            draw.text((bbox[0], bbox[1] - 20), label, fill=color)
        
        # Save annotated image
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        annotated_path = os.path.join('UPLOAD_FOLDER', f"{base_name}_annotated.jpg")
        pil_image.save(annotated_path)
        
        return annotated_path
        
    except Exception as e:
        print(f"Error creating annotated image: {str(e)}")
        return None

def compare_damages(initial_damages, return_damages):
    """
    Compare initial and return damages to find new damages
    """
    new_damages = []
    
    # Simple approach: assume all return damages are new for now
    # You can implement more sophisticated matching logic based on position and type
    for return_damage in return_damages:
        is_new = True
        
        # Check if this damage existed initially (simple overlap check)
        for initial_damage in initial_damages:
            if (return_damage['class'] == initial_damage['class'] and 
                bbox_overlap(return_damage['bbox'], initial_damage['bbox']) > 0.3):
                is_new = False
                break
        
        if is_new:
            new_damages.append(return_damage)
    
    return new_damages

def bbox_overlap(bbox1, bbox2):
    """
    Calculate overlap ratio between two bounding boxes
    """
    try:
        # Calculate intersection
        x_left = max(bbox1[0], bbox2[0])
        y_top = max(bbox1[1], bbox2[1])
        x_right = min(bbox1[2], bbox2[2])
        y_bottom = min(bbox1[3], bbox2[3])
        
        if x_right <= x_left or y_bottom <= y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate areas
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
        
    except:
        return 0.0

@app.route('/create_rental', methods=['POST'])
def create_rental():
    try:
        if rentals_collection is None:
            return jsonify({'error': 'Database connection failed'}), 500
        
        # Get form data
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        location = request.form.get('location')
        pickup_datetime = request.form.get('pickup_datetime')
        return_datetime = request.form.get('return_datetime')
        vehicle_type = request.form.get('vehicle_type')
        vehicle_number = request.form.get('vehicle_number')
        
        # Validate required fields
        if not all([first_name, last_name, location, pickup_datetime, return_datetime, vehicle_type, vehicle_number]):
            return jsonify({'error': 'All fields are required'}), 400
        
        # Define possible views
        views = ['front', 'back', 'left_side', 'right_side', 'roof']
        uploaded_views = {}
        damage_reports = {}
        reference = str(uuid.uuid4())[:8].upper()  # Generate reference upfront
        
        # Process each view (up to 5)
        uploaded_count = 0
        for view in views:
            if view in request.files:
                file = request.files[view]
                if file.filename != '':
                    if uploaded_count >= 5:
                        return jsonify({'error': 'Maximum 5 views can be uploaded'}), 400
                    # Save uploaded photo
                    filename = f"{reference}_{view}_initial.jpg"
                    file_path = os.path.join('UPLOAD_FOLDER', filename)
                    file.save(file_path)
                    
                    # Analyze damage for this view using parallel models
                    damage_report = analyze_damage_with_parallel_models(file_path)
                    
                    uploaded_views[view] = {
                        'filename': filename,
                        'path': file_path
                    }
                    damage_reports[view] = damage_report
                    uploaded_count += 1
        
        # Check if at least one view is uploaded
        if not uploaded_views:
            return jsonify({'error': 'At least one car view photo must be uploaded'}), 400
        
        # Create rental document
        rental_document = {
            'reference': reference,
            'first_name': first_name,
            'last_name': last_name,
            'location': location,
            'pickup_datetime': pickup_datetime,
            'return_datetime': return_datetime,
            'vehicle_type': vehicle_type,
            'vehicle_number': vehicle_number,
            'initial_views': uploaded_views,
            'initial_damage_reports': damage_reports,
            'created_at': datetime.now(),
            'updated_at': datetime.now(),
            'status': 'active',
            'returned': False,
            'analysis_models': list(models.keys())  # Record which models were available
        }
        
        # Insert into MongoDB
        result = rentals_collection.insert_one(rental_document)
        
        if result.inserted_id:
            print(f"✅ Rental created with ID: {result.inserted_id}")
            return jsonify({
                'reference': reference,
                'initial_damage_reports': damage_reports,
                'uploaded_views': list(uploaded_views.keys()),
                'message': 'Rental created successfully',
                'rental_id': str(result.inserted_id),
                'models_used': list(models.keys())
            })
        else:
            return jsonify({'error': 'Failed to create rental'}), 500
        
    except Exception as e:
        print(f"Error creating rental: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/fetch_rental', methods=['POST'])
def fetch_rental():
    try:
        if rentals_collection is None:
            return jsonify({'error': 'Database connection failed'}), 500
        
        data = request.get_json()
        reference = data.get('reference')
        first_name = data.get('first_name')
        last_name = data.get('last_name')
        
        if not all([reference, first_name, last_name]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Find rental in MongoDB
        rental = rentals_collection.find_one({
            'reference': reference,
            'first_name': {'$regex': f'^{first_name}$', '$options': 'i'},  # Case insensitive
            'last_name': {'$regex': f'^{last_name}$', '$options': 'i'}
        })
        
        if not rental:
            return jsonify({'error': 'Rental not found or customer details do not match'}), 404
        
        # Return rental information (excluding sensitive data)
        return jsonify({
            'reference': rental['reference'],
            'first_name': rental['first_name'],
            'last_name': rental['last_name'],
            'location': rental['location'],
            'pickup_datetime': rental['pickup_datetime'],
            'return_datetime': rental['return_datetime'],
            'vehicle_type': rental['vehicle_type'],
            'vehicle_number': rental['vehicle_number'],
            'status': rental.get('status', 'active'),
            'returned': rental.get('returned', False),
            'rental_id': str(rental['_id'])
        })
        
    except Exception as e:
        print(f"Error fetching rental: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/submit_return/<reference>', methods=['POST'])
def submit_return(reference):
    try:
        if rentals_collection is None:
            return jsonify({'error': 'Database connection failed'}), 500
        
        # Find rental in MongoDB
        rental = rentals_collection.find_one({'reference': reference})
        
        if not rental:
            return jsonify({'error': 'Rental not found'}), 404
        
        if rental.get('returned', False):
            return jsonify({'error': 'This rental has already been returned'}), 400
        
        # Define possible views
        views = ['front', 'back', 'left_side', 'right_side', 'roof']
        return_views = {}
        return_damage_reports = {}
        uploaded_count = 0
        
        # Process each return view (up to 5)
        for view in views:
            if view in request.files:
                file = request.files[view]
                if file.filename != '':
                    if uploaded_count >= 5:
                        return jsonify({'error': 'Maximum 5 views can be uploaded'}), 400
                    # Save return photo
                    filename = f"{reference}_{view}_return.jpg"
                    file_path = os.path.join('UPLOAD_FOLDER', filename)
                    file.save(file_path)
                    
                    # Analyze return damage using parallel models
                    damage_report = analyze_damage_with_parallel_models(file_path)
                    
                    return_views[view] = {
                        'filename': filename,
                        'path': file_path
                    }
                    return_damage_reports[view] = damage_report
                    uploaded_count += 1
        
        # Check if at least one return view is uploaded
        if not return_views:
            return jsonify({'error': 'At least one return photo must be uploaded'}), 400
        
        # Compare damages view by view
        initial_damage_reports = rental.get('initial_damage_reports', {})
        comparison_report = compare_all_views(initial_damage_reports, return_damage_reports)
        
        # Create comprehensive report
        report = {
            'rental_reference': reference,
            'return_datetime': datetime.now().isoformat(),
            'views_comparison': comparison_report,
            'summary': generate_damage_summary(comparison_report),
            'uploaded_return_views': list(return_views.keys()),
            'initial_views_available': list(initial_damage_reports.keys()),
            'analysis_models_used': list(models.keys())
        }
        
        # Update rental document in MongoDB
        update_data = {
            '$set': {
                'status': 'returned',
                'returned': True,
                'return_datetime_actual': datetime.now(),
                'return_views': return_views,
                'return_damage_reports': return_damage_reports,
                'final_report': report,
                'updated_at': datetime.now()
            }
        }
        
        result = rentals_collection.update_one(
            {'reference': reference},
            update_data
        )
        
        if result.modified_count > 0:
            print(f"✅ Rental {reference} marked as returned")
            return jsonify({
                'report': report,
                'message': 'Return processed successfully using parallel model analysis'
            })
        else:
            return jsonify({'error': 'Failed to update rental status'}), 500
        
    except Exception as e:
        print(f"Error processing return: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

def get_comparison_status(view, initial_reports, return_reports, new_damages):
    """
    Get status for a specific view comparison
    """
    if view not in initial_reports and view not in return_reports:
        return 'not_available'
    elif view not in initial_reports:
        return 'initial_missing'
    elif view not in return_reports:
        return 'return_missing'
    elif len(new_damages) > 0:
        return 'new_damage_detected'
    else:
        return 'no_new_damage'
    
def compare_all_views(initial_reports, return_reports):
    """
    Compare damages across all available views, handling missing images
    """
    comparison = {}
    
    # Define all possible views
    views = ['front', 'back', 'left_side', 'right_side', 'roof']
    
    for view in views:
        initial_damages = initial_reports.get(view, {}).get('damages', [])
        return_damages = return_reports.get(view, {}).get('damages', [])
        
        # Compare damages if both images are available
        new_damages = []
        if initial_reports.get(view) and return_reports.get(view):
            new_damages = compare_damages(initial_damages, return_damages)
        
        comparison[view] = {
            'initial_available': view in initial_reports,
            'return_available': view in return_reports,
            'initial_damages': initial_damages,
            'return_damages': return_damages,
            'new_damages': new_damages,
            'has_new_damage': len(new_damages) > 0,
            'damage_count': {
                'initial': len(initial_damages),
                'return': len(return_damages),
                'new': len(new_damages)
            },
            'initial_annotated_image': initial_reports.get(view, {}).get('annotated_image'),
            'return_annotated_image': return_reports.get(view, {}).get('annotated_image'),
            'comparison_status': get_comparison_status(view, initial_reports, return_reports, new_damages),
            'note': (
                'Comparison completed using parallel models' if view in initial_reports and view in return_reports
                else 'Missing initial image' if view in return_reports
                else 'Missing return image' if view in initial_reports
                else 'No images provided'
            )
        }
    
    return comparison    

def generate_damage_summary(comparison_report):
    """
    Generate overall summary from all views comparison
    """
    total_views_compared = 0
    total_new_damages = 0
    views_with_new_damage = []
    views_status = {}
    missing_initial_views = []
    missing_return_views = []
    
    for view, data in comparison_report.items():
        views_status[view] = data['comparison_status']
        if data['initial_available'] and data['return_available']:
            total_views_compared += 1
            total_new_damages += data['damage_count']['new']
            if data['has_new_damage']:
                views_with_new_damage.append(view)
        elif data['initial_available'] and not data['return_available']:
            missing_return_views.append(view)
        elif data['return_available'] and not data['initial_available']:
            missing_initial_views.append(view)
    
    return {
        'total_views_available': len(comparison_report),
        'total_views_compared': total_views_compared,
        'total_new_damages': total_new_damages,
        'views_with_new_damage': views_with_new_damage,
        'has_any_new_damage': len(views_with_new_damage) > 0,
        'missing_initial_views': missing_initial_views,
        'missing_return_views': missing_return_views,
        'views_status': views_status,
        'damage_severity': categorize_damage_severity(total_new_damages, views_with_new_damage),
        'analysis_method': 'parallel_models',
        'models_used': list(models.keys()),
        'note': (
            f"Analyzed using {len([m for m in models.values() if m is not None])} parallel models. "
            f"Compared {total_views_compared} views. "
            f"Missing initial images for {len(missing_initial_views)} views: {missing_initial_views}. "
            f"Missing return images for {len(missing_return_views)} views: {missing_return_views}."
        )
    }

def categorize_damage_severity(total_damages, affected_views):
    """
    Categorize overall damage severity
    """
    if total_damages == 0:
        return 'no_damage'
    elif total_damages <= 2 and len(affected_views) == 1:
        return 'minor'
    elif total_damages <= 5 and len(affected_views) <= 2:
        return 'moderate'
    else:
        return 'major'

@app.route('/rentals', methods=['GET'])
def get_all_rentals():
    """Get all rentals (optional admin endpoint)"""
    try:
        if rentals_collection is None:
            return jsonify({'error': 'Database connection failed'}), 500
        
        # Get query parameters
        status = request.args.get('status')  # active, returned
        limit = int(request.args.get('limit', 50))
        skip = int(request.args.get('skip', 0))
        
        # Build query
        query = {}
        if status:
            query['status'] = status
        
        # Fetch rentals
        rentals = list(rentals_collection.find(query)
                      .sort('created_at', -1)  # Sort by newest first
                      .skip(skip)
                      .limit(limit))
        
        # Convert ObjectId to string
        for rental in rentals:
            rental['_id'] = str(rental['_id'])
            if 'created_at' in rental:
                rental['created_at'] = rental['created_at'].isoformat()
            if 'updated_at' in rental:
                rental['updated_at'] = rental['updated_at'].isoformat()
            if 'return_datetime_actual' in rental:
                rental['return_datetime_actual'] = rental['return_datetime_actual'].isoformat()
        
        return jsonify({
            'rentals': rentals,
            'count': len(rentals),
            'total': rentals_collection.count_documents(query)
        })
        
    except Exception as e:
        print(f"Error fetching rentals: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/rental/<reference>', methods=['GET'])
def get_rental_details(reference):
    """Get detailed rental information"""
    try:
        if rentals_collection is None:
            return jsonify({'error': 'Database connection failed'}), 500
        
        rental = rentals_collection.find_one({'reference': reference})
        
        if not rental:
            return jsonify({'error': 'Rental not found'}), 404
        
        # Convert ObjectId and datetime objects
        rental['_id'] = str(rental['_id'])
        if 'created_at' in rental:
            rental['created_at'] = rental['created_at'].isoformat()
        if 'updated_at' in rental:
            rental['updated_at'] = rental['updated_at'].isoformat()
        if 'return_datetime_actual' in rental:
            rental['return_datetime_actual'] = rental['return_datetime_actual'].isoformat()
        
        return jsonify(rental)
        
    except Exception as e:
        print(f"Error fetching rental details: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    mongo_status = "connected" if client is not None else "disconnected"
    models_status = {name: "loaded" if model_data is not None else "failed" 
                    for name, model_data in models.items()}
    
    return jsonify({
        'status': 'healthy',
        'message': 'Car rental API is running with parallel model analysis',
        'mongodb': mongo_status,
        'database': DB_NAME,
        'collection': COLLECTION_NAME,
        'models': models_status,
        'total_models': len(models),
        'active_models': len([m for m in models.values() if m is not None])
    })

@app.route('/models/status', methods=['GET'])
def get_models_status():
    """Get detailed information about loaded models"""
    models_info = {}
    
    with models_lock:
        for model_name, model_data in models.items():
            if model_data is not None:
                config = model_data['config']
                models_info[model_name] = {
                    'status': 'loaded',
                    'workspace': config['workspace'],
                    'project': config['project'],
                    'version': config['version'],
                    'confidence_threshold': config['confidence'],
                    'overlap_threshold': config['overlap']
                }
            else:
                models_info[model_name] = {
                    'status': 'failed',
                    'error': 'Model failed to load'
                }
    
    return jsonify({
        'models': models_info,
        'total_configured': len(MODELS_CONFIG),
        'total_loaded': len([m for m in models.values() if m is not None]),
        'parallel_processing': True
    })

@app.route('/models/reload', methods=['POST'])
def reload_models():
    """Reload all Roboflow models"""
    try:
        print("🔄 Reloading Roboflow models...")
        initialize_roboflow_models()
        
        loaded_count = len([m for m in models.values() if m is not None])
        
        return jsonify({
            'message': f'Models reloaded successfully. {loaded_count}/{len(MODELS_CONFIG)} models loaded.',
            'models_status': {name: "loaded" if model_data is not None else "failed" 
                            for name, model_data in models.items()}
        })
        
    except Exception as e:
        print(f"Error reloading models: {str(e)}")
        return jsonify({'error': 'Failed to reload models'}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get enhanced rental statistics including view-specific data and model performance"""
    try:
        if rentals_collection is None:
            return jsonify({'error': 'Database connection failed'}), 500
        
        total_rentals = rentals_collection.count_documents({})
        active_rentals = rentals_collection.count_documents({'status': 'active'})
        returned_rentals = rentals_collection.count_documents({'status': 'returned'})
        
        # Enhanced damage statistics
        damaged_returns = rentals_collection.count_documents({
            'final_report.summary.has_any_new_damage': True
        })
        
        # View-specific statistics
        view_stats = {}
        views = ['front', 'back', 'left_side', 'right_side', 'roof']
        
        for view in views:
            view_damaged = rentals_collection.count_documents({
                f'final_report.views_comparison.{view}.has_new_damage': True
            })
            view_stats[view] = {
                'damaged_returns': view_damaged,
                'damage_rate': round((view_damaged / returned_rentals * 100), 2) if returned_rentals > 0 else 0
            }
        
        # Damage severity distribution
        severity_stats = {}
        for severity in ['minor', 'moderate', 'major']:
            count = rentals_collection.count_documents({
                'final_report.summary.damage_severity': severity
            })
            severity_stats[severity] = count
        
        # Model performance statistics
        parallel_analysis_count = rentals_collection.count_documents({
            'final_report.summary.analysis_method': 'parallel_models'
        })
        
        return jsonify({
            'total_rentals': total_rentals,
            'active_rentals': active_rentals,
            'returned_rentals': returned_rentals,
            'damaged_returns': damaged_returns,
            'clean_returns': returned_rentals - damaged_returns,
            'damage_rate': round((damaged_returns / returned_rentals * 100), 2) if returned_rentals > 0 else 0,
            'view_statistics': view_stats,
            'damage_severity_distribution': severity_stats,
            'model_analysis': {
                'parallel_analysis_used': parallel_analysis_count,
                'models_configured': len(MODELS_CONFIG),
                'models_currently_loaded': len([m for m in models.values() if m is not None]),
                'parallel_processing_rate': round((parallel_analysis_count / returned_rentals * 100), 2) if returned_rentals > 0 else 0
            }
        })
        
    except Exception as e:
        print(f"Error fetching stats: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/test_models/<path:filename>', methods=['POST'])
def test_models_on_image(filename):
    """Test parallel model analysis on a specific uploaded image"""
    try:
        image_path = os.path.join('UPLOAD_FOLDER', filename)
        
        if not os.path.exists(image_path):
            return jsonify({'error': 'Image not found'}), 404
        
        # Run parallel analysis
        result = analyze_damage_with_parallel_models(image_path)
        
        return jsonify({
            'test_result': result,
            'image_tested': filename,
            'models_used': result.get('models_used', []),
            'message': 'Test analysis completed successfully'
        })
        
    except Exception as e:
        print(f"Error testing models: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚗 Starting Enhanced Car Rental API with Parallel Model Analysis...")
    print(f"📁 Upload folder: {'UPLOAD_FOLDER'}")
    print(f"🗄️  Database: {DB_NAME}")
    print(f"📊 Collection: {COLLECTION_NAME}")
    print(f"🤖 Models configured: {len(MODELS_CONFIG)}")
    
    for i, config in enumerate(MODELS_CONFIG, 1):
        print(f"   Model {i}: {config['workspace']}/{config['project']} v{config['version']}")
    
    if client is not None:
        print("✅ MongoDB connection successful")
    else:
        print("❌ MongoDB connection failed - some features may not work")
    
    loaded_models = len([m for m in models.values() if m is not None])
    print(f"🤖 Loaded {loaded_models}/{len(MODELS_CONFIG)} Roboflow models")
    
    if loaded_models > 0:
        print("✅ Parallel model analysis ready")
    else:
        print("⚠️  No models loaded - using fallback analysis")
    
    print("🚀 Server starting on http://0.0.0.0:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)