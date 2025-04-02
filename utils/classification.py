import numpy as np
import cv2
import pandas as pd
from utils.detection import detect_cards_from_image

def predict_color(shape_image):
    """
    Predict the dominant color in the shape image using HSV thresholds.

    Args:
        shape_image (numpy.ndarray): Image in BGR format.

    Returns:
        str: Predicted color ('green', 'purple', or 'red').
    """
    hsv_image = cv2.cvtColor(shape_image, cv2.COLOR_BGR2HSV)
    # Define HSV ranges for colors.
    green_lower = np.array([40, 50, 50])
    green_upper = np.array([80, 255, 255])
    purple_lower = np.array([120, 50, 50])
    purple_upper = np.array([160, 255, 255])
    red_lower1 = np.array([0, 50, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 50, 50])
    red_upper2 = np.array([180, 255, 255])
    # Create color masks.
    green_mask = cv2.inRange(hsv_image, green_lower, green_upper)
    purple_mask = cv2.inRange(hsv_image, purple_lower, purple_upper)
    red_mask1 = cv2.inRange(hsv_image, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv_image, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    # Count non-zero pixels.
    green_count = cv2.countNonZero(green_mask)
    purple_count = cv2.countNonZero(purple_mask)
    red_count = cv2.countNonZero(red_mask)
    color_counts = {'green': green_count, 'purple': purple_count, 'red': red_count}
    return max(color_counts, key=color_counts.get)

def predict_card_features(card_image, shape_detection_model, fill_model, shape_model, box):
    """
    Predict card features (count, color, fill, shape) from a card image.

    Args:
        card_image (numpy.ndarray): Card image in BGR format.
        shape_detection_model: YOLO model for detecting shapes on the card.
        fill_model: Keras model for fill classification.
        shape_model: Keras model for shape classification.
        box (array-like): Bounding box of the card.

    Returns:
        dict: Dictionary containing predicted card features.
    """
    # Detect shapes on the card.
    shape_results = shape_detection_model(card_image)
    card_height, card_width = card_image.shape[:2]
    card_area = card_width * card_height

    # Filter out detections that are too small (less than 3% of card area).
    filtered_boxes = []
    for detected_box in shape_results[0].boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = detected_box.astype(int)
        shape_width = x2 - x1
        shape_height = y2 - y1
        shape_area = shape_width * shape_height
        if shape_area > 0.03 * card_area:
            filtered_boxes.append(detected_box)

    count = min(len(filtered_boxes), 3)
    color_labels = []
    fill_labels = []
    shape_labels = []

    for shape_box in filtered_boxes:
        shape_box = shape_box.astype(int)
        # Crop the card image to the shape.
        shape_img = card_image[shape_box[1]:shape_box[3], shape_box[0]:shape_box[2]]
        # Preprocess for classification.
        fill_input_shape = fill_model.input_shape[1:3]
        shape_input_shape = shape_model.input_shape[1:3]
        fill_img = cv2.resize(shape_img, fill_input_shape) / 255.0
        shape_img_resized = cv2.resize(shape_img, shape_input_shape) / 255.0
        fill_img = np.expand_dims(fill_img, axis=0)
        shape_img_resized = np.expand_dims(shape_img_resized, axis=0)
        # Obtain predictions.
        fill_pred = fill_model.predict(fill_img)
        shape_pred = shape_model.predict(shape_img_resized)
        color_labels.append(predict_color(shape_img))
        fill_labels.append(['empty', 'full', 'striped'][np.argmax(fill_pred)])
        shape_labels.append(['diamond', 'oval', 'squiggle'][np.argmax(shape_pred)])

    if count > 0:
        color_label = max(set(color_labels), key=color_labels.count)
        fill_label = max(set(fill_labels), key=fill_labels.count)
        shape_label = max(set(shape_labels), key=shape_labels.count)
    else:
        color_label = 'unknown'
        fill_label = 'unknown'
        shape_label = 'unknown'

    return {
        'count': count,
        'color': color_label,
        'fill': fill_label,
        'shape': shape_label,
        'box': box
    }

def classify_cards_from_board_image(board_image, card_detection_model, shape_detection_model, fill_model, shape_model):
    """
    Classify cards on a board image by detecting each card and predicting its features.

    Args:
        board_image (numpy.ndarray): The corrected board image.
        card_detection_model: YOLO model for card detection.
        shape_detection_model: YOLO model for shape detection.
        fill_model: Keras model for fill classification.
        shape_model: Keras model for shape classification.

    Returns:
        pandas.DataFrame: DataFrame containing card features.
    """
    cards = detect_cards_from_image(board_image, card_detection_model)
    card_data = []
    for card_image, box in cards:
        features = predict_card_features(card_image, shape_detection_model, fill_model, shape_model, box)
        card_data.append({
            "Count": features['count'],
            "Color": features['color'],
            "Fill": features['fill'],
            "Shape": features['shape'],
            "Coordinates": f"{box[0]}, {box[1]}, {box[2]}, {box[3]}"
        })
    return pd.DataFrame(card_data)
