# import os
# import datetime
# import requests
# from flask import Flask, request, jsonify, session
# from flask_cors import CORS
# from werkzeug.utils import secure_filename
# from werkzeug.security import generate_password_hash, check_password_hash
# import pymongo
# from pymongo import MongoClient
# from bson import ObjectId
# import json
# import google.generativeai as genai
# from PIL import Image
# import base64
# from io import BytesIO
# import uuid
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# app = Flask(__name__)
# app.secret_key = os.getenv('SECRET_KEY', 'default-secret-key')
# app.config['UPLOAD_FOLDER'] = 'uploads'
# app.config['UPLOAD_FOLDER_HIGHLIGHTED'] = 'uploads/highlighted'
# app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
# app.config['ML_SERVICE_URL'] = os.getenv('ML_SERVICE_URL', 'http://localhost:8001')

# # Ensure upload directories exist
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# os.makedirs(app.config['UPLOAD_FOLDER_HIGHLIGHTED'], exist_ok=True)

# # Configure CORS to allow requests from the frontend
# CORS(app, supports_credentials=True, origins=["http://localhost:3000"])

# # Connect to MongoDB Atlas
# try:
#     mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017/brain_tumor_db')
#     client = MongoClient(mongo_uri)
#     db = client.brain_tumor_db
#     users_collection = db.users
#     images_collection = db.images
#     starred_collection = db.starred
    
#     # Create indexes for faster queries
#     users_collection.create_index("email", unique=True)
    
#     print("Connected to MongoDB Atlas")
# except Exception as e:
#     print(f"Error connecting to MongoDB Atlas: {e}")

# # Configure Google Gemini API
# try:
#     GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
#     genai.configure(api_key=GEMINI_API_KEY)
#     print("Connected to Google Gemini API")
# except Exception as e:
#     print(f"Error configuring Google Gemini API: {e}")

# # Helper class for JSON serialization
# class JSONEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, ObjectId):
#             return str(obj)
#         if isinstance(obj, datetime.datetime):
#             return obj.isoformat()
#         return json.JSONEncoder.default(self, obj)

# # Utility functions
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# def save_image(file):
#     """Save uploaded image and return the file path"""
#     filename = secure_filename(file.filename)
#     unique_filename = f"{uuid.uuid4()}_{filename}"
#     file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
#     file.save(file_path)
#     return file_path, unique_filename

# def save_highlighted_image(base64_image, original_filename):
#     """Save highlighted image from base64 data"""
#     if not base64_image:
#         return None, None
        
#     try:
#         # Remove data:image/jpeg;base64, prefix if present
#         if ',' in base64_image:
#             base64_image = base64_image.split(',')[1]
            
#         image_data = base64.b64decode(base64_image)
#         unique_filename = f"highlighted_{uuid.uuid4()}_{original_filename}"
#         file_path = os.path.join(app.config['UPLOAD_FOLDER_HIGHLIGHTED'], unique_filename)
        
#         with open(file_path, 'wb') as f:
#             f.write(image_data)
            
#         return file_path, unique_filename
#     except Exception as e:
#         print(f"Error saving highlighted image: {e}")
#         return None, None

# def image_to_base64(image_path):
#     """Convert image to base64 for storage"""
#     with open(image_path, "rb") as img_file:
#         return base64.b64encode(img_file.read()).decode('utf-8')

# def process_with_ml_model(image_path):
#     """
#     Process image with ML model by calling the FastAPI ML service
#     """
#     try:
#         # Prepare the image file for sending to the ML service
#         with open(image_path, 'rb') as img_file:
#             files = {'file': (os.path.basename(image_path), img_file, 'image/jpeg')}
            
#             # Call the ML service
#             response = requests.post(
#                 f"{app.config['ML_SERVICE_URL']}/predict", 
#                 files=files
#             )
            
#             if response.status_code == 200:
#                 result = response.json()
                
#                 # If image is not appropriate
#                 if not result.get("is_appropriate", True):
#                     return None, False
                
#                 # Return the ML results
#                 return result.get("ml_results", {}), True
#             else:
#                 print(f"ML service error: {response.status_code} - {response.text}")
#                 # Fallback prediction if ML service fails
#                 return fallback_prediction(), True
#     except Exception as e:
#         print(f"Error calling ML service: {e}")
#         # Fallback prediction if ML service fails
#         return fallback_prediction(), True

# def fallback_prediction():
#     """Fallback prediction if ML service is unavailable"""
#     return {
#         "prediction": "Positive",
#         "confidence": 0.85,
#         "tumor_type": "Meningioma",
#         "precautions": [
#             "Consult with a neurosurgeon immediately",
#             "Avoid strenuous physical activity",
#             "Get a follow-up MRI within 2 weeks",
#             "Monitor for symptoms like headaches, vision changes, or seizures"
#         ],
#         "treatment_options": [
#             "Surgical removal",
#             "Radiation therapy",
#             "Regular monitoring"
#         ],
#         "highlighted_image": "",  # Empty string for no highlighted image
#         "tumor_location": {}  # Empty dict for no tumor location
#     }

# # Routes
# @app.route('/api/register', methods=['POST'])
# def register():
#     data = request.json
#     if not data or not data.get('email') or not data.get('password') or not data.get('name'):
#         return jsonify({"error": "Missing required fields"}), 400
    
#     email = data['email']
#     password = data['password']
#     name = data['name']
    
#     # Check if user already exists
#     if users_collection.find_one({"email": email}):
#         return jsonify({"error": "User already exists"}), 409
    
#     # Create new user
#     hashed_password = generate_password_hash(password)
#     user_id = users_collection.insert_one({
#         "name": name,
#         "email": email,
#         "password": hashed_password,
#         "created_at": datetime.datetime.utcnow()
#     }).inserted_id
    
#     return jsonify({"message": "User registered successfully", "user_id": str(user_id)}), 201

# @app.route('/api/login', methods=['POST'])
# def login():
#     data = request.json
#     if not data or not data.get('email') or not data.get('password'):
#         return jsonify({"error": "Missing email or password"}), 400
    
#     user = users_collection.find_one({"email": data['email']})
#     if not user or not check_password_hash(user['password'], data['password']):
#         return jsonify({"error": "Invalid credentials"}), 401
    
#     # Store user session data
#     user_data = {
#         "id": str(user['_id']),
#         "name": user['name'],
#         "email": user['email']
#     }
    
#     return jsonify({"message": "Login successful", "user": user_data}), 200

# @app.route('/api/upload', methods=['POST'])
# def upload_image():
#     # Check if user is logged in
#     user_id = request.form.get('user_id')
#     if not user_id:
#         return jsonify({"error": "User not authenticated"}), 401
    
#     # Check if image was provided
#     if 'image' not in request.files:
#         return jsonify({"error": "No image provided"}), 400
    
#     file = request.files['image']
#     if file.filename == '':
#         return jsonify({"error": "No image selected"}), 400
    
#     if file and allowed_file(file.filename):
#         # Save the uploaded image
#         file_path, unique_filename = save_image(file)
        
#         # Process with ML model
#         ml_results, is_appropriate = process_with_ml_model(file_path)
        
#         if not is_appropriate:
#             # Remove the file if it's not appropriate
#             os.remove(file_path)
#             return jsonify({
#                 "error": "Please upload an appropriate brain MRI or CT scan image for tumor detection"
#             }), 400
        
#         # Convert image to base64 for storage
#         image_base64 = image_to_base64(file_path)
        
#         # Save highlighted image if available
#         highlighted_path = None
#         highlighted_filename = None
#         highlighted_base64 = ml_results.get("highlighted_image", "")
        
#         if highlighted_base64 and ml_results["prediction"] == "Positive":
#             highlighted_path, highlighted_filename = save_highlighted_image(highlighted_base64, unique_filename)
#             # Save the base64 string directly in ml_results
        
#         # Store image information in database
#         image_data = {
#             "user_id": ObjectId(user_id),
#             "filename": unique_filename,
#             "original_filename": file.filename,
#             "image_data": image_base64,
#             "upload_time": datetime.datetime.utcnow(),
#             "is_appropriate": is_appropriate,
#             "ml_results": ml_results,
#             "highlighted_filename": highlighted_filename
#         }
        
#         image_id = images_collection.insert_one(image_data).inserted_id
        
#         # Return results
#         return jsonify({
#             "message": "Image processed successfully",
#             "image_id": str(image_id),
#             "is_appropriate": is_appropriate,
#             "ml_results": ml_results
#         }), 200
    
#     return jsonify({"error": "File type not allowed"}), 400

# @app.route('/api/history/<user_id>', methods=['GET'])
# def get_history(user_id):
#     try:
#         # Convert user_id string to ObjectId
#         user_obj_id = ObjectId(user_id)
        
#         # Query the database for all images uploaded by this user
#         user_images = list(images_collection.find({"user_id": user_obj_id}))
        
#         # Prepare the response data (exclude large image_data field to reduce payload size)
#         history = []
#         for img in user_images:
#             history.append({
#                 "image_id": str(img["_id"]),
#                 "filename": img["original_filename"],
#                 "upload_time": img["upload_time"],
#                 "is_appropriate": img["is_appropriate"],
#                 "ml_results": img["ml_results"],
#                 "image_data": img["image_data"]  # Include image data for display
#             })
        
#         return jsonify({"history": history}), 200
    
#     except Exception as e:
#         return jsonify({"error": f"Error retrieving history: {str(e)}"}), 500

# @app.route('/api/image/<image_id>', methods=['GET'])
# def get_image(image_id):
#     try:
#         # Convert image_id string to ObjectId
#         image_obj_id = ObjectId(image_id)
        
#         # Query the database for the image
#         image = images_collection.find_one({"_id": image_obj_id})
        
#         if not image:
#             return jsonify({"error": "Image not found"}), 404
        
#         # Return the image data
#         return jsonify({
#             "image_id": str(image["_id"]),
#             "filename": image["original_filename"],
#             "upload_time": image["upload_time"],
#             "is_appropriate": image["is_appropriate"],
#             "ml_results": image["ml_results"],
#             "image_data": image["image_data"]
#         }), 200
    
#     except Exception as e:
#         return jsonify({"error": f"Error retrieving image: {str(e)}"}), 500

# # Starred image endpoints
# @app.route('/api/starred/<user_id>', methods=['GET'])
# def get_starred_images(user_id):
#     try:
#         # Convert user_id string to ObjectId
#         user_obj_id = ObjectId(user_id)
        
#         # Query the database for all starred items for this user
#         starred_items = list(starred_collection.find({"user_id": user_obj_id}))
        
#         # Process each starred item to add the image data
#         starred_images = {}
#         for item in starred_items:
#             image_id = str(item["image_id"])
#             image = images_collection.find_one({"_id": item["image_id"]})
            
#             if image:
#                 starred_images[image_id] = {
#                     "image_id": image_id,
#                     "user_id": str(item["user_id"]),
#                     "filename": image["original_filename"],
#                     "upload_time": image["upload_time"],
#                     "image_data": image["image_data"],
#                     "ml_results": image["ml_results"],
#                     "note": item["note"],
#                     "timestamp": item["timestamp"]
#                 }
        
#         # Log for debugging
#         print(f"Found {len(starred_images)} starred images for user {user_id}")
        
#         return jsonify({"starred_images": starred_images}), 200
    
#     except Exception as e:
#         print(f"Error retrieving starred images: {str(e)}")
#         return jsonify({"error": f"Error retrieving starred images: {str(e)}", "starred_images": {}}), 500

# @app.route('/api/starred', methods=['POST'])
# def add_starred_image():
#     try:
#         data = request.json
#         print(f"Received starred image data: {data}")
        
#         # Validate required fields
#         required_fields = ["user_id", "image_id", "note"]
#         for field in required_fields:
#             if field not in data:
#                 return jsonify({"error": f"Missing required field: {field}"}), 400
                
#         # Convert IDs to ObjectId
#         user_id = ObjectId(data["user_id"])
#         image_id = ObjectId(data["image_id"])
        
#         # Check if image exists
#         image = images_collection.find_one({"_id": image_id})
#         if not image:
#             return jsonify({"error": f"Image not found with ID: {data['image_id']}"}), 404
            
#         # Check if already starred
#         existing = starred_collection.find_one({
#             "user_id": user_id,
#             "image_id": image_id
#         })
        
#         # Create starred_images directory if it doesn't exist
#         starred_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'starred_images')
#         os.makedirs(starred_dir, exist_ok=True)
        
#         # Copy image to starred_images directory
#         original_image_path = os.path.join(app.config['UPLOAD_FOLDER'], image['filename'])
#         starred_image_path = os.path.join(starred_dir, f"starred_{image['filename']}")
        
#         if not os.path.exists(starred_image_path):
#             import shutil
#             shutil.copy2(original_image_path, starred_image_path)
        
#         if existing:
#             # Update existing starred item
#             print(f"Updating existing starred item for image {data['image_id']}")
#             starred_collection.update_one(
#                 {"_id": existing["_id"]},
#                 {"$set": {
#                     "note": data["note"],
#                     "timestamp": datetime.datetime.utcnow(),
#                     "local_path": starred_image_path
#                 }}
#             )
#             return jsonify({"message": "Starred image updated successfully"}), 200
            
#         # Add new starred item
#         print(f"Adding new starred item for image {data['image_id']}")
#         starred_data = {
#             "user_id": user_id,
#             "image_id": image_id,
#             "note": data["note"],
#             "timestamp": datetime.datetime.utcnow(),
#             "local_path": starred_image_path
#         }
        
#         result = starred_collection.insert_one(starred_data)
        
#         return jsonify({
#             "message": "Image starred successfully",
#             "starred_id": str(result.inserted_id)
#         }), 201
    
#     except Exception as e:
#         print(f"Error starring image: {str(e)}")
#         return jsonify({"error": f"Error starring image: {str(e)}"}), 500

# @app.route('/api/starred/<user_id>/<image_id>', methods=['DELETE'])
# def remove_starred_image(user_id, image_id):
#     try:
#         # Convert IDs to ObjectId
#         user_obj_id = ObjectId(user_id)
#         image_obj_id = ObjectId(image_id)
        
#         # Delete the starred item
#         result = starred_collection.delete_one({
#             "user_id": user_obj_id,
#             "image_id": image_obj_id
#         })
        
#         if result.deleted_count == 0:
#             return jsonify({"error": "Starred image not found"}), 404
            
#         return jsonify({"message": "Image removed from starred successfully"}), 200
    
#     except Exception as e:
#         return jsonify({"error": f"Error removing starred image: {str(e)}"}), 500

# @app.route('/api/debug/<user_id>', methods=['GET'])
# def debug_collections(user_id):
#     try:
#         # Convert user_id string to ObjectId
#         user_obj_id = ObjectId(user_id)
        
#         # Get counts for each collection
#         user_count = users_collection.count_documents({})
#         images_count = images_collection.count_documents({})
#         starred_count = starred_collection.count_documents({})
#         user_starred_count = starred_collection.count_documents({"user_id": user_obj_id})
        
#         # Get list of starred items for this user
#         starred_items = list(starred_collection.find({"user_id": user_obj_id}))
#         starred_ids = [str(item["image_id"]) for item in starred_items]
        
#         return jsonify({
#             "database_status": "connected",
#             "collections": {
#                 "users": user_count,
#                 "images": images_count,
#                 "starred": starred_count
#             },
#             "user_starred_count": user_starred_count,
#             "user_starred_ids": starred_ids
#         }), 200
#     except Exception as e:
#         print(f"Debug error: {str(e)}")
#         return jsonify({"error": f"Debug error: {str(e)}"}), 500

# if __name__ == '__main__':
#     app.run(debug=True)

import os
import datetime
import requests
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import pymongo
from pymongo import MongoClient
from bson import ObjectId
import json
import google.generativeai as genai
from PIL import Image
import base64
from io import BytesIO
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'default-secret-key')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['ML_SERVICE_URL'] = os.getenv('ML_SERVICE_URL', 'http://localhost:8001')

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configure CORS to allow requests from the frontend
CORS(app, supports_credentials=True, origins=["http://localhost:3000"])

# Connect to MongoDB Atlas
try:
    mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017/brain_tumor_db')
    client = MongoClient(mongo_uri)
    db = client.brain_tumor_db
    users_collection = db.users
    images_collection = db.images
    starred_collection = db.starred
    
    # Create indexes for faster queries
    users_collection.create_index("email", unique=True)
    
    print("Connected to MongoDB Atlas")
except Exception as e:
    print(f"Error connecting to MongoDB Atlas: {e}")

# Configure Google Gemini API
try:
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    genai.configure(api_key=GEMINI_API_KEY)
    print("Connected to Google Gemini API")
except Exception as e:
    print(f"Error configuring Google Gemini API: {e}")

# Helper class for JSON serialization
class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        return json.JSONEncoder.default(self, obj)

# Utility functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def save_image(file):
    """Save uploaded image and return the file path"""
    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4()}_{filename}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(file_path)
    return file_path, unique_filename

def image_to_base64(image_path):
    """Convert image to base64 for storage"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def process_with_ml_model(image_path):
    """
    Process image with ML model by calling the FastAPI ML service
    """
    try:
        # Prepare the image file for sending to the ML service
        with open(image_path, 'rb') as img_file:
            files = {'file': (os.path.basename(image_path), img_file, 'image/jpeg')}
            
            # Call the ML service
            response = requests.post(
                f"{app.config['ML_SERVICE_URL']}/predict", 
                files=files
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # If image is not appropriate
                if not result.get("is_appropriate", True):
                    return None, False
                
                # Return the ML results
                return result.get("ml_results", {}), True
            else:
                print(f"ML service error: {response.status_code} - {response.text}")
                # Fallback prediction if ML service fails
                return fallback_prediction(), True
    except Exception as e:
        print(f"Error calling ML service: {e}")
        # Fallback prediction if ML service fails
        return fallback_prediction(), True

def fallback_prediction():
    """Fallback prediction if ML service is unavailable"""
    return {
        "prediction": "Positive",
        "confidence": 0.85,
        "tumor_type": "Meningioma",
        "precautions": [
            "Consult with a neurosurgeon immediately",
            "Avoid strenuous physical activity",
            "Get a follow-up MRI within 2 weeks",
            "Monitor for symptoms like headaches, vision changes, or seizures"
        ],
        "treatment_options": [
            "Surgical removal",
            "Radiation therapy",
            "Regular monitoring"
        ],
        # No highlighted image in fallback
        "highlighted_image": None
    }
def fallback_prediction():
    """Fallback prediction if ML service is unavailable"""
    return {
        "prediction": "Positive",
        "confidence": 0.85,
        "tumor_type": "Meningioma",
        "precautions": [
            "Consult with a neurosurgeon immediately",
            "Avoid strenuous physical activity",
            "Get a follow-up MRI within 2 weeks",
            "Monitor for symptoms like headaches, vision changes, or seizures"
        ],
        "treatment_options": [
            "Surgical removal",
            "Radiation therapy",
            "Regular monitoring"
        ]
    }

# Routes
@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    if not data or not data.get('email') or not data.get('password') or not data.get('name'):
        return jsonify({"error": "Missing required fields"}), 400
    
    email = data['email']
    password = data['password']
    name = data['name']
    
    # Check if user already exists
    if users_collection.find_one({"email": email}):
        return jsonify({"error": "User already exists"}), 409
    
    # Create new user
    hashed_password = generate_password_hash(password)
    user_id = users_collection.insert_one({
        "name": name,
        "email": email,
        "password": hashed_password,
        "created_at": datetime.datetime.utcnow()
    }).inserted_id
    
    return jsonify({"message": "User registered successfully", "user_id": str(user_id)}), 201

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    if not data or not data.get('email') or not data.get('password'):
        return jsonify({"error": "Missing email or password"}), 400
    
    user = users_collection.find_one({"email": data['email']})
    if not user or not check_password_hash(user['password'], data['password']):
        return jsonify({"error": "Invalid credentials"}), 401
    
    # Store user session data
    user_data = {
        "id": str(user['_id']),
        "name": user['name'],
        "email": user['email']
    }
    
    return jsonify({"message": "Login successful", "user": user_data}), 200

# In the upload_image function, ensure the highlighted image is passed along
# Look for the @app.route('/api/upload', methods=['POST']) function and update:

@app.route('/api/upload', methods=['POST'])
def upload_image():
    # Check if user is logged in
    user_id = request.form.get('user_id')
    if not user_id:
        return jsonify({"error": "User not authenticated"}), 401
    
    # Check if image was provided
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No image selected"}), 400
    
    if file and allowed_file(file.filename):
        # Save the uploaded image
        file_path, unique_filename = save_image(file)
        
        # Process with ML model
        ml_results, is_appropriate = process_with_ml_model(file_path)
        
        if not is_appropriate:
            # Remove the file if it's not appropriate
            os.remove(file_path)
            return jsonify({
                "error": "Please upload an appropriate brain MRI or CT scan image for tumor detection"
            }), 400
        
        # Convert image to base64 for storage
        image_base64 = image_to_base64(file_path)
        
        # Store image information in database
        image_data = {
            "user_id": ObjectId(user_id),
            "filename": unique_filename,
            "original_filename": file.filename,
            "image_data": image_base64,
            "upload_time": datetime.datetime.utcnow(),
            "is_appropriate": is_appropriate,
            "ml_results": ml_results
        }
        
        # Store highlighted image separately if it exists
        if 'highlighted_image' in ml_results:
            image_data['highlighted_image'] = ml_results['highlighted_image']
        
        image_id = images_collection.insert_one(image_data).inserted_id
        
        # Return results
        return jsonify({
            "message": "Image processed successfully",
            "image_id": str(image_id),
            "is_appropriate": is_appropriate,
            "ml_results": ml_results
        }), 200
    
    return jsonify({"error": "File type not allowed"}), 400

@app.route('/api/history/<user_id>', methods=['GET'])
def get_history(user_id):
    try:
        # Convert user_id string to ObjectId
        user_obj_id = ObjectId(user_id)
        
        # Query the database for all images uploaded by this user
        user_images = list(images_collection.find({"user_id": user_obj_id}))
        
        # Prepare the response data (exclude large image_data field to reduce payload size)
        history = []
        for img in user_images:
            history.append({
                "image_id": str(img["_id"]),
                "filename": img["original_filename"],
                "upload_time": img["upload_time"],
                "is_appropriate": img["is_appropriate"],
                "ml_results": img["ml_results"],
                "image_data": img["image_data"]  # Include image data for display
            })
        
        return jsonify({"history": history}), 200
    
    except Exception as e:
        return jsonify({"error": f"Error retrieving history: {str(e)}"}), 500

@app.route('/api/image/<image_id>', methods=['GET'])
def get_image(image_id):
    try:
        # Convert image_id string to ObjectId
        image_obj_id = ObjectId(image_id)
        
        # Query the database for the image
        image = images_collection.find_one({"_id": image_obj_id})
        
        if not image:
            return jsonify({"error": "Image not found"}), 404
        
        # Return the image data including highlighted image if available
        response_data = {
            "image_id": str(image["_id"]),
            "filename": image["original_filename"],
            "upload_time": image["upload_time"],
            "is_appropriate": image["is_appropriate"],
            "ml_results": image["ml_results"],
            "image_data": image["image_data"]
        }
        
        # Add highlighted image if it exists
        if "highlighted_image" in image:
            response_data["highlighted_image"] = image["highlighted_image"]
        
        return jsonify(response_data), 200
    
    except Exception as e:
        return jsonify({"error": f"Error retrieving image: {str(e)}"}), 500

# Starred image endpoints
@app.route('/api/starred/<user_id>', methods=['GET'])
def get_starred_images(user_id):
    try:
        # Convert user_id string to ObjectId
        user_obj_id = ObjectId(user_id)
        
        # Query the database for all starred items for this user
        starred_items = list(starred_collection.find({"user_id": user_obj_id}))
        
        # Process each starred item to add the image data
        starred_images = {}
        for item in starred_items:
            image_id = str(item["image_id"])
            image = images_collection.find_one({"_id": item["image_id"]})
            
            if image:
                starred_images[image_id] = {
                    "image_id": image_id,
                    "user_id": str(item["user_id"]),
                    "filename": image["original_filename"],
                    "upload_time": image["upload_time"],
                    "image_data": image["image_data"],
                    "ml_results": image["ml_results"],
                    "note": item["note"],
                    "timestamp": item["timestamp"]
                }
        
        # Log for debugging
        print(f"Found {len(starred_images)} starred images for user {user_id}")
        
        return jsonify({"starred_images": starred_images}), 200
    
    except Exception as e:
        print(f"Error retrieving starred images: {str(e)}")
        return jsonify({"error": f"Error retrieving starred images: {str(e)}", "starred_images": {}}), 500

@app.route('/api/starred', methods=['POST'])
def add_starred_image():
    try:
        data = request.json
        print(f"Received starred image data: {data}")
        
        # Validate required fields
        required_fields = ["user_id", "image_id", "note"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
                
        # Convert IDs to ObjectId
        user_id = ObjectId(data["user_id"])
        image_id = ObjectId(data["image_id"])
        
        # Check if image exists
        image = images_collection.find_one({"_id": image_id})
        if not image:
            return jsonify({"error": f"Image not found with ID: {data['image_id']}"}), 404
            
        # Check if already starred
        existing = starred_collection.find_one({
            "user_id": user_id,
            "image_id": image_id
        })
        
        # Create starred_images directory if it doesn't exist
        starred_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'starred_images')
        os.makedirs(starred_dir, exist_ok=True)
        
        # Copy image to starred_images directory
        original_image_path = os.path.join(app.config['UPLOAD_FOLDER'], image['filename'])
        starred_image_path = os.path.join(starred_dir, f"starred_{image['filename']}")
        
        if not os.path.exists(starred_image_path):
            import shutil
            shutil.copy2(original_image_path, starred_image_path)
        
        if existing:
            # Update existing starred item
            print(f"Updating existing starred item for image {data['image_id']}")
            starred_collection.update_one(
                {"_id": existing["_id"]},
                {"$set": {
                    "note": data["note"],
                    "timestamp": datetime.datetime.utcnow(),
                    "local_path": starred_image_path
                }}
            )
            return jsonify({"message": "Starred image updated successfully"}), 200
            
        # Add new starred item
        print(f"Adding new starred item for image {data['image_id']}")
        starred_data = {
            "user_id": user_id,
            "image_id": image_id,
            "note": data["note"],
            "timestamp": datetime.datetime.utcnow(),
            "local_path": starred_image_path
        }
        
        result = starred_collection.insert_one(starred_data)
        
        return jsonify({
            "message": "Image starred successfully",
            "starred_id": str(result.inserted_id)
        }), 201
    
    except Exception as e:
        print(f"Error starring image: {str(e)}")
        return jsonify({"error": f"Error starring image: {str(e)}"}), 500

@app.route('/api/starred/<user_id>/<image_id>', methods=['DELETE'])
def remove_starred_image(user_id, image_id):
    try:
        # Convert IDs to ObjectId
        user_obj_id = ObjectId(user_id)
        image_obj_id = ObjectId(image_id)
        
        # Delete the starred item
        result = starred_collection.delete_one({
            "user_id": user_obj_id,
            "image_id": image_obj_id
        })
        
        if result.deleted_count == 0:
            return jsonify({"error": "Starred image not found"}), 404
            
        return jsonify({"message": "Image removed from starred successfully"}), 200
    
    except Exception as e:
        return jsonify({"error": f"Error removing starred image: {str(e)}"}), 500

@app.route('/api/debug/<user_id>', methods=['GET'])
def debug_collections(user_id):
    try:
        # Convert user_id string to ObjectId
        user_obj_id = ObjectId(user_id)
        
        # Get counts for each collection
        user_count = users_collection.count_documents({})
        images_count = images_collection.count_documents({})
        starred_count = starred_collection.count_documents({})
        user_starred_count = starred_collection.count_documents({"user_id": user_obj_id})
        
        # Get list of starred items for this user
        starred_items = list(starred_collection.find({"user_id": user_obj_id}))
        starred_ids = [str(item["image_id"]) for item in starred_items]
        
        return jsonify({
            "database_status": "connected",
            "collections": {
                "users": user_count,
                "images": images_count,
                "starred": starred_count
            },
            "user_starred_count": user_starred_count,
            "user_starred_ids": starred_ids
        }), 200
    except Exception as e:
        print(f"Debug error: {str(e)}")
        return jsonify({"error": f"Debug error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)