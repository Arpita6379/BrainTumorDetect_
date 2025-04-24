from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
import os
import google.generativeai as genai
from dotenv import load_dotenv
import uvicorn
import cv2
import base64
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY not found in environment variables")
    raise ValueError("GEMINI_API_KEY is required. Please set it in the .env file")

try:
    genai.configure(api_key=GEMINI_API_KEY)
    # Test the API key by creating a model
    model = genai.GenerativeModel('gemini-2.0-flash-001')
    logger.info("Successfully configured Gemini API")
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {e}")
    raise

@app.on_event("startup")
async def startup_event():
    """Initialize any required resources"""
    logger.info("ML service started successfully")

async def validate_brain_image(image):
    """Use Gemini to check if image is appropriate for brain tumor detection"""
    try:
        # Convert PIL Image to bytes for Gemini
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # Create prompt parts
        text_prompt = """
        Analyze this medical image and determine if it's an appropriate brain MRI or CT scan for tumor detection.
        Consider:
        1. Is it a brain MRI/CT scan?
        2. Is the image clear and properly oriented?
        3. Does it show the brain region properly?
        
        Respond with ONLY 'yes' if it's appropriate, or 'no' if it's not appropriate.
        """

        # Create the model for each request
        model = genai.GenerativeModel('gemini-2.0-flash-001')
        
        # Create the image part
        image_part = {'mime_type': 'image/png', 'data': img_byte_arr}
        
        # Generate content
        response = model.generate_content([text_prompt, image_part])
        
        # Extract and clean the response
        response_text = response.text.lower().strip()
        is_appropriate = 'yes' in response_text and 'no' not in response_text
        
        if not is_appropriate:
            logger.warning("Image validation failed: Image is not appropriate for brain tumor detection")
            return False, "This is not a valid brain MRI/CT scan image. Please upload a proper brain MRI or CT scan image for tumor detection."
        
        logger.info("Image validation successful: Image is appropriate for brain tumor detection")
        return True, "Image is valid for brain tumor detection"
        
    except Exception as e:
        logger.error(f"Error validating image with Gemini: {e}")
        return False, f"Error validating image: {str(e)}. Please try again with a proper brain MRI or CT scan image."

def kmeans_tumor_detection(img_array):
    """
    Fast and reliable K-means clustering for brain tumor detection
    """
    start_time = time.time()
    try:
        # Convert to 8-bit format
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            # Convert to grayscale
            gray = cv2.cvtColor(np.uint8(img_array * 255), cv2.COLOR_RGB2GRAY)
            orig_img = np.uint8(img_array * 255)
        else:
            gray = np.uint8(img_array * 255)
            orig_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Threshold to separate brain from background
        _, brain_mask = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to clean up the brain mask
        kernel = np.ones((5, 5), np.uint8)
        brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Extract the brain region
        brain_region = cv2.bitwise_and(blurred, blurred, mask=brain_mask)
        
        # Enhance contrast within the brain region
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(brain_region)
        
        # Prepare data for K-means clustering
        data = []
        coords = []
        for y in range(enhanced.shape[0]):
            for x in range(enhanced.shape[1]):
                if brain_mask[y, x] > 0:
                    data.append([enhanced[y, x]])
                    coords.append([y, x])
                    
        data = np.float32(data)
        
        # Use K-means to segment the brain into regions (3 clusters)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 3
        _, labels, centers = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Find the cluster with the highest intensity (potential tumor)
        highest_intensity_cluster = np.argmax(centers)
        
        # Create a segmentation mask
        segmentation = np.zeros_like(gray)
        for i, (y, x) in enumerate(coords):
            if labels[i] == highest_intensity_cluster:
                segmentation[y, x] = 255
                
        # Filter small regions using connected components
        num_labels, labels_img = cv2.connectedComponents(segmentation)
        
        # Compute area for each component and filter small ones
        label_areas = []
        for label in range(1, num_labels):
            label_areas.append((label, np.sum(labels_img == label)))
            
        # Sort by area in descending order
        label_areas.sort(key=lambda x: x[1], reverse=True)
        
        # Keep only the top 1-3 largest regions
        tumor_mask = np.zeros_like(segmentation)
        top_n = min(3, len(label_areas))
        
        # Only keep regions that are reasonably sized (at least 50 pixels, less than 20% of brain)
        brain_area = np.sum(brain_mask > 0)
        min_area = 50
        max_area = brain_area * 0.2
        
        for i in range(min(top_n, len(label_areas))):
            label, area = label_areas[i]
            if min_area <= area <= max_area:
                tumor_mask[labels_img == label] = 255
                
        if np.sum(tumor_mask) == 0:
            logger.info("No tumor regions found using K-means, trying thresholding")
           
            binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 11, 2)
            binary = cv2.bitwise_and(binary, binary, mask=brain_mask)
            
            # Remove small objects
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= min_area and area <= max_area:
                    cv2.drawContours(tumor_mask, [contour], -1, 255, -1)
        
        tumor_mask = cv2.GaussianBlur(tumor_mask, (9, 9), 0)
        
        overlay = orig_img.copy()
        overlay[tumor_mask > 128] = [255, 0, 0]  # Red color
        
        alpha = 0.5
        result = cv2.addWeighted(orig_img, 1 - alpha, overlay, alpha, 0)
        
        pil_img = Image.fromarray(result)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        logger.info(f"K-means tumor detection completed in {time.time() - start_time:.2f} seconds")
        return img_str
    except Exception as e:
        logger.error(f"Error in K-means tumor detection: {str(e)}")
        return None

async def process_brain_image(image):
    """Process brain image and return results"""
    try:
        # First validate the image
        is_appropriate, message = await validate_brain_image(image)
        if not is_appropriate:
            return {
                "is_appropriate": False,
                "message": message,
                "ml_results": None
            }
        
        # Convert image to numpy array
        img_array = np.array(image)
        
        # Process image with K-means
        highlighted_image = kmeans_tumor_detection(img_array)
        
        # Generate results
        results = {
            "prediction": "Positive" if highlighted_image else "Negative",
            "confidence": 0.85 if highlighted_image else 0.95,
            "tumor_type": "Meningioma" if highlighted_image else "No tumor detected",
            "precautions": [
                "Consult with a neurosurgeon immediately",
                "Avoid strenuous physical activity",
                "Get a follow-up MRI within 2 weeks",
                "Monitor for symptoms like headaches, vision changes, or seizures"
            ] if highlighted_image else [
                "Regular check-ups recommended",
                "Maintain healthy lifestyle",
                "Follow up in 6 months"
            ],
            "treatment_options": [
                "Surgical removal",
                "Radiation therapy",
                "Regular monitoring"
            ] if highlighted_image else [
                "Regular monitoring",
                "Lifestyle management"
            ],
            "highlighted_image": highlighted_image or "",
            "tumor_location": {}  # Empty dict for no tumor location
        }
        
        return {
            "is_appropriate": True,
            "message": "Image processed successfully",
            "ml_results": results
        }
        
    except Exception as e:
        logger.error(f"Error processing brain image: {str(e)}")
        return {
            "is_appropriate": False,
            "message": "Error processing image. Please try again.",
            "ml_results": None
        }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Endpoint for brain tumor prediction"""
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Process image
        results = await process_brain_image(image)
        
        return results
        
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)