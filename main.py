from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from aruco_triangle_detection import process_image_with_calibration, explain_triangle_with_gemini
import logging
import io
import os
import google.generativeai as genai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure Gemini API
os.environ["GEMINI_API_KEY"] = "AIzaSyBMZshYv40LxHWivZdoRQfR1Z6aGZddzC8"
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Initialize FastAPI app
app = FastAPI(
    title="Triangle AI Detection API",
    description="API for detecting and analyzing triangles using ArUco markers",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins in development
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Triangle Detection API is running"}

@app.post("/analyze-triangle")
async def analyze_triangle(file: UploadFile = File(...)):
    """
    Analyze a triangle in an uploaded image.
    
    Parameters:
    - file: Image file containing a triangle with ArUco markers
    
    Returns:
    - Dictionary containing triangle analysis results
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Process image
        logger.info(f"Processing image: {file.filename}")
        result_image, angles, distances, classification, area, perimeter = process_image_with_calibration(image)
        
        # Get Gemini explanation
        try:
            generation_config = {
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 2048,
            }

            model = genai.GenerativeModel(
                model_name="gemini-pro",
                generation_config=generation_config,
            )

            prompt = f"""
            You are a friendly math teacher explaining a triangle to middle school students (grades 6-8).
            Here's a triangle we measured in class:
            Angles: {angles['A']:.1f}°, {angles['B']:.1f}°, {angles['C']:.1f}°
            Sides: {distances['A-B']:.1f} cm, {distances['B-C']:.1f} cm, {distances['C-A']:.1f} cm
            Type: {classification}
            Area: {area:.1f} cm²
            Perimeter: {perimeter:.1f} cm

            Write a short, explanation about this triangle focusing on its angles, sides, area, and perimeter. 
            Include a simple real-world example where we might find this type of triangle.
            Use friendly, conversational language that a middle school student would understand easily.
            Avoid using complex mathematical terms or making it sound like a textbook.

            start like "Here this triangle is a "
            """

            response = model.generate_content(prompt)
            explanation = response.text
            logger.info("Generated Gemini explanation successfully")
        except Exception as e:
            logger.error(f"Error generating Gemini explanation: {str(e)}")
            explanation = "Sorry, I couldn't generate an explanation at this time."
        
        # Encode result image
        _, img_encoded = cv2.imencode('.jpg', result_image)
        img_base64 = img_encoded.tobytes()
        
        # Prepare response
        response = {
            "angles": {k: float(v) for k, v in angles.items()},
            "distances": {k: float(v) for k, v in distances.items()},
            "classification": classification,
            "area": float(area),
            "perimeter": float(perimeter),
            "processed_image": img_base64,
            "explanation": explanation
        }
        
        logger.info(f"Successfully processed image: {file.filename}")
        return response
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
