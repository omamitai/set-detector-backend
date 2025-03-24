from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2
import time
import base64
import tempfile
import os
import os.path
import sys
import logging
import tensorflow as tf
from tensorflow.keras.models import load_model
import torch
from ultralytics import YOLO
from pathlib import Path
import traceback
import uvicorn
from PIL import Image
import io
import pandas as pd
from itertools import combinations
import asyncio
from contextlib import asynccontextmanager
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("set-detector")

# Global variables for models
detector_card = None
detector_shape = None
model_shape = None
model_fill = None
models_loaded = False

# Define model paths
BASE_DIR = Path("models")
CHAR_PATH = BASE_DIR / "Characteristics" / "11022025"
SHAPE_PATH = BASE_DIR / "Shape" / "15052024"
CARD_PATH = BASE_DIR / "Card" / "16042024"


# Load models on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup
    try:
        load_models()
        logger.info("Application started with models loaded")
    except Exception as e:
        logger.error(f"Failed to load models at startup: {str(e)}")
        logger.error(traceback.format_exc())
    
    yield
    
    # Optional cleanup on shutdown
    logger.info("Shutting down application")


# Initialize FastAPI app with lifespan
app = FastAPI(
    lifespan=lifespan, 
    title="SET Game Detector API",
    description="API for detecting SET cards and valid SETs in images",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify actual origins
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Response models
class CardFeature(BaseModel):
    position: int
    features: Dict[str, str]


class SetFound(BaseModel):
    set_id: int
    cards: List[CardFeature]


class DetectionResponse(BaseModel):
    image: str  # Base64 encoded image
    sets_found: List[SetFound]
    all_cards: List[CardFeature]
    processing_time: float


# Load models with improved error handling
def load_models():
    global detector_card, detector_shape, model_shape, model_fill, models_loaded

    if models_loaded:
        logger.info("Models already loaded, skipping")
        return

    logger.info("Loading models...")
    
    # Verify model files exist
    model_files = [
        (CHAR_PATH / "shape_model.keras", "Shape classification model"),
        (CHAR_PATH / "fill_model.keras", "Fill classification model"),
        (SHAPE_PATH / "best.pt", "Shape detection model"),
        (CARD_PATH / "best.pt", "Card detection model")
    ]
    
    for file_path, description in model_files:
        if not file_path.exists():
            raise FileNotFoundError(f"Missing model file: {description} at {file_path}")
    
    try:
        # Load Keras models for shape and fill classification
        model_shape = load_model(str(CHAR_PATH / "shape_model.keras"))
        logger.info("Shape model loaded")
        
        model_fill = load_model(str(CHAR_PATH / "fill_model.keras"))
        logger.info("Fill model loaded")

        # Load YOLO detection models
        detector_shape = YOLO(str(SHAPE_PATH / "best.pt"))
        detector_shape.conf = 0.5
        logger.info("Shape detector loaded")
        
        detector_card = YOLO(str(CARD_PATH / "best.pt"))
        detector_card.conf = 0.5
        logger.info("Card detector loaded")

        # Use GPU if available
        if torch.cuda.is_available():
            detector_card.to("cuda")
            detector_shape.to("cuda")
            logger.info("YOLO models loaded on GPU")
        else:
            logger.info("YOLO models loaded on CPU")

        models_loaded = True
        logger.info("All models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Failed to load models: {str(e)}")


# Utility Functions
def verify_and_rotate_image(board_image: np.ndarray, card_detector: YOLO) -> Tuple[np.ndarray, bool]:
    """
    Checks if the detected cards are oriented primarily vertically or horizontally.
    If they're vertical, rotates the board_image 90 degrees clockwise for consistent processing.
    Returns (possibly_rotated_image, was_rotated_flag).
    """
    detection = card_detector(board_image)
    boxes = detection[0].boxes.xyxy.cpu().numpy().astype(int)
    if boxes.size == 0:
        return board_image, False

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]

    # Rotate if average height > average width
    if np.mean(heights) > np.mean(widths):
        return cv2.rotate(board_image, cv2.ROTATE_90_CLOCKWISE), True
    else:
        return board_image, False


def restore_orientation(img: np.ndarray, was_rotated: bool) -> np.ndarray:
    """
    Restores original orientation if the image was previously rotated.
    """
    if was_rotated:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def predict_color(img_bgr: np.ndarray) -> str:
    """
    Rough color classification using HSV thresholds to differentiate 'red', 'green', 'purple'.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, np.array([40, 50, 50]), np.array([80, 255, 255]))
    mask_purple = cv2.inRange(hsv, np.array([120, 50, 50]), np.array([160, 255, 255]))

    # Red can wrap around hue=0, so we combine both ends
    mask_red1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
    mask_red2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    counts = {
        "green": cv2.countNonZero(mask_green),
        "purple": cv2.countNonZero(mask_purple),
        "red": cv2.countNonZero(mask_red),
    }
    return max(counts, key=counts.get)


def detect_cards(board_img: np.ndarray, card_detector: YOLO) -> List[Tuple[np.ndarray, List[int]]]:
    """
    Runs YOLO on the board_img to detect card bounding boxes.
    Returns a list of (card_image, [x1, y1, x2, y2]) for each detected card.
    """
    result = card_detector(board_img)
    boxes = result[0].boxes.xyxy.cpu().numpy().astype(int)
    detected_cards = []

    for x1, y1, x2, y2 in boxes:
        detected_cards.append((board_img[y1:y2, x1:x2], [x1, y1, x2, y2]))
    return detected_cards


def predict_card_features(
    card_img: np.ndarray,
    shape_detector: YOLO,
    fill_model: tf.keras.Model,
    shape_model: tf.keras.Model,
    card_box: List[int],
    card_index: int
) -> Dict:
    """
    Predicts the 'count', 'color', 'fill', 'shape' features for a single card.
    It uses a shape_detector YOLO model to locate shapes, then passes them to fill_model and shape_model.
    """
    # Detect shapes on the card
    shape_detections = shape_detector(card_img)
    c_h, c_w = card_img.shape[:2]
    card_area = c_w * c_h

    # Filter out spurious shape detections
    shape_boxes = []
    for coords in shape_detections[0].boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = coords.astype(int)
        if (x2 - x1) * (y2 - y1) > 0.03 * card_area:
            shape_boxes.append([x1, y1, x2, y2])

    if not shape_boxes:
        return {
            'number': '0',
            'color': 'unknown',
            'shading': 'unknown',
            'shape': 'unknown',
            'position': card_index,
            'box': card_box
        }

    fill_input_size = fill_model.input_shape[1:3]
    shape_input_size = shape_model.input_shape[1:3]
    fill_imgs = []
    shape_imgs = []
    color_candidates = []

    # Prepare each detected shape region for classification
    for sb in shape_boxes:
        sx1, sy1, sx2, sy2 = sb
        shape_crop = card_img[sy1:sy2, sx1:sx2]

        fill_crop = cv2.resize(shape_crop, fill_input_size) / 255.0
        shape_crop_resized = cv2.resize(shape_crop, shape_input_size) / 255.0

        fill_imgs.append(fill_crop)
        shape_imgs.append(shape_crop_resized)
        color_candidates.append(predict_color(shape_crop))

    # Use model.predict in batch mode for better performance
    fill_preds = fill_model.predict(np.array(fill_imgs), batch_size=len(fill_imgs), verbose=0)
    shape_preds = shape_model.predict(np.array(shape_imgs), batch_size=len(shape_imgs), verbose=0)

    fill_labels = ['empty', 'full', 'striped']
    shape_labels = ['diamond', 'oval', 'squiggle']

    fill_result = [fill_labels[np.argmax(fp)] for fp in fill_preds]
    shape_result = [shape_labels[np.argmax(sp)] for sp in shape_preds]

    # Take the most common color/fill/shape across all shape detections for the card
    final_color = max(set(color_candidates), key=color_candidates.count)
    final_fill = max(set(fill_result), key=fill_result.count)
    final_shape = max(set(shape_result), key=shape_result.count)

    # Convert to the expected format
    return {
        'number': str(len(shape_boxes)),
        'color': final_color,
        'shading': final_fill,
        'shape': final_shape,
        'position': card_index,
        'box': card_box
    }


def classify_cards_on_board(
    board_img: np.ndarray,
    card_detector: YOLO,
    shape_detector: YOLO,
    fill_model: tf.keras.Model,
    shape_model: tf.keras.Model
) -> List[Dict]:
    """
    Detects cards on the board, then classifies each card's features.
    Returns a list of cards with their features and positions.
    """
    detected_cards = detect_cards(board_img, card_detector)
    card_data = []

    for i, (card_img, box) in enumerate(detected_cards):
        card_feats = predict_card_features(card_img, shape_detector, fill_model, shape_model, box, i)
        card_data.append(card_feats)

    return card_data


def valid_set(cards: List[Dict]) -> bool:
    """
    Checks if the given 3 cards collectively form a valid SET.
    """
    # Convert feature names to match the API format
    feature_map = {
        "number": "Count",
        "color": "Color",
        "shading": "Fill",
        "shape": "Shape"
    }
    
    for api_feature, internal_feature in feature_map.items():
        values = set()
        # Skip unknown values
        if any(card[api_feature] == 'unknown' for card in cards):
            return False
            
        for card in cards:
            values.add(card[api_feature])
            
        # Must be all same or all different
        if len(values) not in (1, 3):
            return False
            
    return True


def locate_all_sets(cards: List[Dict]) -> List[Dict]:
    """
    Finds all possible SETs from the card list.
    Each SET is a dictionary with info about the cards in the set.
    """
    found_sets = []
    for i, combo in enumerate(combinations(range(len(cards)), 3)):
        card_combo = [cards[j] for j in combo]
        if valid_set(card_combo):
            found_sets.append({
                'set_id': i,
                'cards': [
                    {
                        'position': card['position'],
                        'features': {
                            'number': card['number'],
                            'color': card['color'],
                            'shape': card['shape'],
                            'shading': card['shading']
                        }
                    }
                    for card in card_combo
                ]
            })
    return found_sets


def draw_detected_sets(board_img: np.ndarray, sets_detected: List[Dict], all_cards: List[Dict]) -> np.ndarray:
    """
    Annotates the board image with visually appealing bounding boxes for each detected SET.
    Each SET is drawn in a different color with enhanced iOS-style indicators.
    """
    # Enhanced SET-themed colors (BGR format)
    set_colors = [
        (68, 68, 239),    # Red
        (89, 199, 16),    # Green
        (246, 92, 139),   # Pink
        (0, 149, 255),    # Orange
        (222, 82, 175),   # Purple
        (85, 45, 255)     # Hot Pink
    ]
    
    # Enhanced styling parameters
    base_thickness = 4
    base_expansion = 5

    # Create a copy to avoid modifying the original
    result_img = board_img.copy()

    # Create a position to card mapping
    card_map = {card['position']: card for card in all_cards}

    for idx, single_set in enumerate(sets_detected):
        color = set_colors[idx % len(set_colors)]
        thickness = base_thickness + (idx % 3)
        expansion = base_expansion + 10 * (idx % 3)

        for i, card_info in enumerate(single_set["cards"]):
            pos = card_info["position"]
            if pos not in card_map:
                continue
                
            x1, y1, x2, y2 = card_map[pos]["box"]
            
            # Expand the bounding box slightly
            x1e = max(0, x1 - expansion)
            y1e = max(0, y1 - expansion)
            x2e = min(result_img.shape[1], x2 + expansion)
            y2e = min(result_img.shape[0], y2 + expansion)

            # Draw premium rounded rectangle with gradient effect
            cv2.rectangle(result_img, (x1e, y1e), (x2e, y2e), color, thickness, cv2.LINE_AA)
            
            # Add iOS-style rounded corners with thicker lines
            corner_length = min(30, (x2e - x1e) // 4)
            corner_thickness = thickness + 2  # Increased thickness for more prominent corners
            
            # Draw all four corners
            # Top-left corner
            cv2.line(result_img, (x1e, y1e + corner_length), (x1e, y1e), color, corner_thickness, cv2.LINE_AA)
            cv2.line(result_img, (x1e, y1e), (x1e + corner_length, y1e), color, corner_thickness, cv2.LINE_AA)
            
            # Top-right corner
            cv2.line(result_img, (x2e - corner_length, y1e), (x2e, y1e), color, corner_thickness, cv2.LINE_AA)
            cv2.line(result_img, (x2e, y1e), (x2e, y1e + corner_length), color, corner_thickness, cv2.LINE_AA)
            
            # Bottom-left corner
            cv2.line(result_img, (x1e, y2e - corner_length), (x1e, y2e), color, corner_thickness, cv2.LINE_AA)
            cv2.line(result_img, (x1e, y2e), (x1e + corner_length, y2e), color, corner_thickness, cv2.LINE_AA)
            
            # Bottom-right corner
            cv2.line(result_img, (x2e - corner_length, y2e), (x2e, y2e), color, corner_thickness, cv2.LINE_AA)
            cv2.line(result_img, (x2e, y2e), (x2e, y2e - corner_length), color, corner_thickness, cv2.LINE_AA)

            # Add subtle inner glow (two concentric rectangles with different opacity)
            inner_color = (
                min(255, color[0] + 40),
                min(255, color[1] + 40),
                min(255, color[2] + 40)
            )
            
            # Inner rectangle (glow effect)
            inner_expansion = expansion - 2
            x1i = max(0, x1 - inner_expansion)
            y1i = max(0, y1 - inner_expansion)
            x2i = min(result_img.shape[1], x2 + inner_expansion)
            y2i = min(result_img.shape[0], y2 + inner_expansion)
            cv2.rectangle(result_img, (x1i, y1i), (x2i, y2i), inner_color, 1, cv2.LINE_AA)

            # Label the first card with Set number in an iOS-style pill badge
            if i == 0:
                text = f"Set {idx+1}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                badge_width = text_size[0] + 16
                badge_height = 28  # Slightly taller for premium look
                badge_x = x1e
                badge_y = y1e - badge_height - 6
                
                # Draw rounded pill-shaped badge with anti-aliasing
                cv2.rectangle(result_img, 
                              (badge_x, badge_y), 
                              (badge_x + badge_width, badge_y + badge_height),
                              color, -1, cv2.LINE_AA)
                
                # Add subtle white inner border to badge for iOS style
                cv2.rectangle(result_img, 
                              (badge_x, badge_y), 
                              (badge_x + badge_width, badge_y + badge_height),
                              (255, 255, 255), 1, cv2.LINE_AA)
                
                # Draw text in center of badge with better positioning
                cv2.putText(
                    result_img,
                    text,
                    (badge_x + 8, badge_y + badge_height - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
    
    return result_img


def optimize_image_size(img_np: np.ndarray, max_dim=800) -> np.ndarray:
    """
    Intelligently resizes an image while preserving aspect ratio.
    """
    height, width = img_np.shape[:2]
    
    # Skip if image is already an appropriate size
    if max(width, height) <= max_dim:
        return img_np
    
    # Calculate the scaling factor
    if width > height:
        new_width = max_dim
        new_height = int(height * (max_dim / width))
    else:
        new_height = max_dim
        new_width = int(width * (max_dim / height))
    
    # Resize the image
    return cv2.resize(img_np, (new_width, new_height), interpolation=cv2.INTER_AREA)


def encode_image_to_base64(img_np: np.ndarray) -> str:
    """
    Encodes a NumPy image array to base64 string.
    """
    _, buffer = cv2.imencode('.jpg', img_np, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buffer).decode('utf-8')


def process_image(image_data: np.ndarray) -> Dict[str, Any]:
    """
    End-to-end pipeline to classify cards on the board and detect valid sets.
    """
    global detector_card, detector_shape, model_shape, model_fill
    
    start_time = time.time()
    
    try:
        # Ensure models are loaded
        if not models_loaded:
            load_models()
            
        # Convert to BGR for OpenCV processing
        if image_data.shape[2] == 3:  # RGB
            img_bgr = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
        else:  # Assume BGR
            img_bgr = image_data
            
        # Resize image for faster processing
        img_bgr = optimize_image_size(img_bgr, max_dim=800)
        
        # 1. Check and fix orientation if needed
        processed_img, was_rotated = verify_and_rotate_image(img_bgr, detector_card)
        
        # 2. Verify that cards are present
        cards = detect_cards(processed_img, detector_card)
        if not cards:
            logger.warning("No cards detected in image")
            # Return empty response with original image
            empty_response = {
                "image": encode_image_to_base64(img_bgr),
                "sets_found": [],
                "all_cards": [],
                "processing_time": time.time() - start_time
            }
            return empty_response
            
        # 3. Classify each card's features
        card_data = classify_cards_on_board(processed_img, detector_card, detector_shape, model_fill, model_shape)
        
        # 4. Find valid SETs
        found_sets = locate_all_sets(card_data)
        
        # 5. Draw SETs on a copy of the image
        if found_sets:
            annotated_img = draw_detected_sets(processed_img.copy(), found_sets, card_data)
        else:
            logger.info("No valid SETs found")
            annotated_img = processed_img.copy()
            
        # 6. Restore original orientation if needed
        final_img = restore_orientation(annotated_img, was_rotated)
        
        # 7. Convert to base64 for response
        img_base64 = encode_image_to_base64(final_img)
        
        # 8. Format all_cards for response
        all_cards_response = [
            {
                "position": card["position"],
                "features": {
                    "number": card["number"],
                    "color": card["color"],
                    "shape": card["shape"],
                    "shading": card["shading"]
                }
            }
            for card in card_data
        ]
        
        processing_time = time.time() - start_time
        logger.info(f"Image processed in {processing_time:.2f}s, found {len(found_sets)} SETs")
        
        # Return response
        return {
            "image": img_base64,
            "sets_found": found_sets,
            "all_cards": all_cards_response,
            "processing_time": processing_time
        }
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


# Root endpoint with API information
@app.get("/", include_in_schema=False)
async def root():
    """
    Root endpoint that returns basic API information
    """
    return {
        "message": "SET Game Detector API", 
        "status": "online", 
        "note": "API-only mode",
        "endpoints": [
            {"path": "/health", "method": "GET", "description": "Health check endpoint"},
            {"path": "/api/detect", "method": "POST", "description": "Detect SETs in uploaded image"}
        ]
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Health check endpoint that returns the API status
    """
    try:
        # Include essential system info
        memory_info = {}
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = {
                "memory_usage_mb": round(process.memory_info().rss / (1024 * 1024), 2),
                "memory_percent": round(process.memory_percent(), 2)
            }
        except ImportError:
            memory_info = {"note": "psutil not available for memory metrics"}
            
        return {
            "status": "healthy",
            "models_loaded": models_loaded,
            "version": "1.0.0",
            "python_version": sys.version,
            "tensorflow_version": tf.__version__,
            "pytorch_version": torch.__version__,
            "system_info": memory_info
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "version": "1.0.0"
        }


# Main API endpoint for SET detection
@app.post("/api/detect", response_model=DetectionResponse)
async def detect_sets(file: UploadFile = File(...)):
    """
    Endpoint to detect SET cards and valid SETs in an uploaded image.
    
    Args:
        file: An uploaded image file containing SET cards
        
    Returns:
        A JSON response containing:
        - image: Base64 encoded image with highlighted SETs
        - sets_found: List of valid SETs detected in the image
        - all_cards: List of all cards detected in the image
        - processing_time: Time taken to process the image
    """
    # Verify file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Only image files are accepted"
        )
        
    # Process the image
    try:
        # Read the file
        contents = await file.read()
        
        # Generate a unique request ID for tracing
        request_id = str(uuid.uuid4())
        logger.info(f"Processing request {request_id}")
        
        # Convert to NumPy array
        img_np = np.array(Image.open(io.BytesIO(contents)).convert('RGB'))
        
        # Process the image (with a timeout)
        result = await asyncio.wait_for(
            asyncio.to_thread(process_image, img_np),
            timeout=60  # 60-second timeout for large or complex images
        )
        
        # Return results
        return result
        
    except asyncio.TimeoutError:
        logger.error(f"Request timed out")
        raise HTTPException(
            status_code=408,
            detail="Processing timed out. Please try again with a clearer image."
        )
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )


# Check if frontend directory exists and conditionally mount static files
frontend_path = Path("frontend/dist")
if frontend_path.exists() and frontend_path.is_dir():
    try:
        app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")
        logger.info(f"Mounted frontend static files from '{frontend_path}'")
    except Exception as e:
        logger.warning(f"Failed to mount frontend directory: {str(e)}")


# Run the server if executed directly
if __name__ == "__main__":
    try:
        # Parse PORT environment variable with fallback
        port = int(os.environ.get("PORT", 8000))
        logger.info(f"Starting server on port {port}")
        
        # Start uvicorn server
        uvicorn.run(
            "main:app", 
            host="0.0.0.0", 
            port=port, 
            reload=False,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)
