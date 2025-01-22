from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import io
from aruco_triangle_detection import process_image_with_calibration

app = FastAPI(
    title="Triangle Detection API",
    description="API for detecting and classifying triangles using ArUco markers and colored points",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze-triangle")
async def analyze_triangle(image: UploadFile = File(...)):
    """
    Analyze a triangle in the uploaded image.
    
    Parameters:
    - image: Image file containing ArUco markers and colored points forming a triangle
    
    Returns:
    - JSON object containing triangle measurements and classification
    """
    try:
        # Read image file
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Process image using our existing function
        image_result, angles, distances, classification = process_image_with_calibration(img)
        
        # Save the result image
        result_path = "result.jpg"
        cv2.imwrite(result_path, image_result)
        
        # Format the response
        response = {
            "angles": {
                "A": round(angles["A"], 2),
                "B": round(angles["B"], 2),
                "C": round(angles["C"], 2)
            },
            "distances": {
                "A-B": round(distances["A-B"], 2),
                "B-C": round(distances["B-C"], 2),
                "C-A": round(distances["C-A"], 2)
            },
            "classification": classification,
            "result_image": result_path
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Welcome endpoint with usage instructions"""
    return {
        "message": "Welcome to the Triangle Detection API",
        "usage": {
            "endpoint": "/analyze-triangle",
            "method": "POST",
            "parameters": {
                "image": "Upload an image file containing ArUco markers and colored points"
            },
            "example_curl": "curl -X POST http://localhost:8000/analyze-triangle -F 'image=@your_image.jpg'"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
