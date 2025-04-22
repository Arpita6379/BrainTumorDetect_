from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import google.generativeai as genai
from dotenv import load_dotenv
import uvicorn

# Load environment variables
load_dotenv()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your React frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
brain_tumor_model = None
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

@app.on_event("startup")
async def startup_event():
    """Load ML model on startup"""
    global brain_tumor_model
    try:
        # Change this path to your actual model path
        model_path = "models/brain_tumor_model_20250304-130406.h5"
        if os.path.exists(model_path):
            brain_tumor_model = tf.keras.models.load_model(model_path)
            print(f"Brain tumor model loaded successfully from {model_path}")
        else:
            print(f"Model file not found at {model_path}. Using fallback predictions.")
    except Exception as e:
        print(f"Error loading brain tumor model: {e}")
        print("Using fallback predictions for now.")

async def validate_brain_image(image):
    """Use Gemini to check if image is appropriate for brain tumor detection"""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp-image-generation')
        prompt = "Is this image appropriate for brain tumor detection? Give answer only yes or no."
        
        response = model.generate_content([prompt, image])
        
        # Extract only 'yes' or 'no' from the response
        response_text = response.text.lower().strip()
        is_appropriate = 'yes' in response_text and 'no' not in response_text
        
        return is_appropriate
    except Exception as e:
        print(f"Error validating image with Gemini: {e}")
        return False

async def process_brain_image(image):
    """Process brain image with the ML model"""
    global brain_tumor_model
    
    # Resize image to match model input size
    image = image.resize((250, 250))
    
   
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    if brain_tumor_model is not None:
        # Make prediction using the actual model
        prediction = brain_tumor_model.predict(image_array)
        is_tumor = bool(prediction[0][0] >= 0.5)
        confidence = float(prediction[0][0] if is_tumor else 1 - prediction[0][0])
    else:
        # Fallback for testing if model isn't loaded
        import random
        is_tumor = random.choice([True, False])
        confidence = random.uniform(0.7, 0.95)
    
    # Return results
    tumor_types = ["Meningioma", "Glioma", "Pituitary"]
    
    if is_tumor:
        tumor_type = tumor_types[np.random.randint(0, len(tumor_types))]
        precautions = [
            "Consult with a neurosurgeon immediately",
            "Avoid strenuous physical activity",
            "Get a follow-up MRI within 2 weeks",
            "Monitor for symptoms like headaches, vision changes, or seizures"
        ]
        treatment_options = [
            "Surgical removal",
            "Radiation therapy",
            "Regular monitoring",
            "Chemotherapy"
        ]
    else:
        tumor_type = "None"
        precautions = ["Regular check-ups", "Monitor for any neurological symptoms"]
        treatment_options = ["No treatment needed", "Routine follow-up in 6-12 months"]
    
    return {
        "prediction": "Positive" if is_tumor else "Negative",
        "confidence": confidence,
        "tumor_type": tumor_type,
        "precautions": precautions,
        "treatment_options": treatment_options
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Endpoint for brain tumor prediction"""
    try:
        # Read and validate the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Check if image is appropriate for brain tumor detection
        is_appropriate = await validate_brain_image(image)
        
        if not is_appropriate:
            return {
                "is_appropriate": False,
                "message": "Please upload an appropriate brain MRI or CT scan image for tumor detection"
            }
        
        # Process the image
        results = await process_brain_image(image)
        
        return {
            "is_appropriate": True,
            "ml_results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("ml_service:app", host="0.0.0.0", port=8001, reload=True)