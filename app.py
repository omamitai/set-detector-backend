from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import os
import traceback
from tensorflow.keras.models import load_model
from ultralytics import YOLO

# Import utility functions
from utils.detection import check_and_rotate_input_image
from utils.classification import classify_cards_from_board_image
from utils.set_finder import find_sets
from utils.drawing import draw_sets_on_image

app = FastAPI(title="Set Game Detector API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variables
shape_model = None
fill_model = None
shape_detection_model = None
card_detection_model = None

@app.on_event("startup")
async def startup_event():
    global shape_model, fill_model, shape_detection_model, card_detection_model
    
    print("Loading models...")
    
    # Load classification models
    shape_model = load_model('./models/Characteristics/11022025/shape_model.keras')
    fill_model = load_model('./models/Characteristics/11022025/fill_model.keras')

    
    # Load YOLO detection models
    shape_detection_model = YOLO('./models/Shape/15052024/best.pt')
    shape_detection_model.yaml = './models/Shape/15052024/data.yaml'
    shape_detection_model.conf = 0.5
    
    card_detection_model = YOLO('./models/Card/16042024/best.pt')
    card_detection_model.yaml = './models/Card/16042024/data.yaml'
    card_detection_model.conf = 0.5
    
    print("Models loaded successfully")

@app.get("/")
async def root():
    return {"message": "Set Game Detector API is running"}

@app.get("/health")
async def health_check():
    if (shape_model is None or fill_model is None or 
        shape_detection_model is None or card_detection_model is None):
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {"status": "healthy"}

@app.post("/detect_sets")
async def detect_sets(file: UploadFile = File(...)):
    if not (file.content_type.startswith('image/')):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        input_image_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Check orientation and rotate if needed
        corrected_image, was_rotated = check_and_rotate_input_image(input_image_cv, card_detection_model)
        
        # Classify cards
        card_df = classify_cards_from_board_image(
            corrected_image, 
            card_detection_model, 
            shape_detection_model, 
            fill_model, 
            shape_model
        )
        
        # Find sets
        sets_found = find_sets(card_df)
        
        # Draw sets on image
        annotated_image = draw_sets_on_image(corrected_image.copy(), sets_found)
        
        # Restore original orientation if needed
        if was_rotated:
            annotated_image = cv2.rotate(annotated_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Return processed image
        success, encoded_image = cv2.imencode('.jpg', annotated_image)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode image")
        
        return Response(content=encoded_image.tobytes(), media_type="image/jpeg")
    
    except Exception as e:
        error_details = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(error_details)  # Log the error
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
