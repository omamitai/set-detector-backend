from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import os
import traceback
import base64
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

def create_message_image(image, message):
    """
    Create an image with a message displayed on it.
    
    Args:
        image (numpy.ndarray): The input image.
        message (str): The message to display.
        
    Returns:
        numpy.ndarray: Image with message.
    """
    result_image = image.copy()
    
    # Get image dimensions
    height, width = result_image.shape[:2]
    
    # Create semi-transparent overlay for better text visibility
    overlay = result_image.copy()
    cv2.rectangle(overlay, (0, height//2 - 40), (width, height//2 + 40), (0, 0, 0), -1)
    
    # Add the overlay with transparency
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, result_image, 1 - alpha, 0, result_image)
    
    # Add text with message
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(1.0, min(width, height) / 500)  # Scale font with image size
    thickness = 2
    text_size = cv2.getTextSize(message, font, font_scale, thickness)[0]
    
    # Calculate position to center the text
    text_x = (width - text_size[0]) // 2
    text_y = height // 2 + text_size[1] // 2
    
    # Draw text
    cv2.putText(result_image, message, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
    
    return result_image

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
        
        # Prepare response data structure
        response_data = {
            "status": "success",
            "cards_detected": 0,
            "sets_found": 0,
            "image_data": None,
            "message": ""
        }
        
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
        
        # Check if any cards were detected
        if card_df.empty:
            message_image = create_message_image(input_image_cv, "No SET Cards Detected")
            success, encoded_image = cv2.imencode('.jpg', message_image)
            if not success:
                raise HTTPException(status_code=500, detail="Failed to encode image")
            
            response_data["status"] = "no_cards"
            response_data["message"] = "No SET Cards Detected"
            response_data["image_data"] = base64.b64encode(encoded_image.tobytes()).decode('utf-8')
            
            return JSONResponse(content=response_data)
        
        # Update cards detected count
        response_data["cards_detected"] = len(card_df)
        
        # Find sets
        sets_found = find_sets(card_df)
        
        # Update sets found count
        response_data["sets_found"] = len(sets_found)
        
        # Check if any sets were found
        if not sets_found:
            message_image = create_message_image(input_image_cv, "No Valid SET Combinations Found")
            success, encoded_image = cv2.imencode('.jpg', message_image)
            if not success:
                raise HTTPException(status_code=500, detail="Failed to encode image")
            
            response_data["status"] = "no_sets"
            response_data["message"] = "No Valid SET Combinations Found"
            response_data["image_data"] = base64.b64encode(encoded_image.tobytes()).decode('utf-8')
            
            return JSONResponse(content=response_data)
        
        # Draw sets on image
        annotated_image = draw_sets_on_image(corrected_image.copy(), sets_found)
        
        # Restore original orientation if needed
        if was_rotated:
            annotated_image = cv2.rotate(annotated_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Encode the processed image
        success, encoded_image = cv2.imencode('.jpg', annotated_image)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode image")
        
        # Add encoded image to response
        response_data["message"] = f"Found {len(sets_found)} SET{'s' if len(sets_found) != 1 else ''}"
        response_data["image_data"] = base64.b64encode(encoded_image.tobytes()).decode('utf-8')
        
        return JSONResponse(content=response_data)
    
    except Exception as e:
        error_details = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(error_details)  # Log the error
        return JSONResponse(
            content={
                "status": "error",
                "message": str(e),
                "cards_detected": 0,
                "sets_found": 0,
                "image_data": None
            },
            status_code=500
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
