import numpy as np
import cv2

def check_and_rotate_input_image(board_image, card_detection_model):
    """
    Checks the orientation of the board image by detecting card bounding boxes.
    If the average detected card height is greater than the average card width,
    rotates the board image 90° clockwise.

    Args:
        board_image (numpy.ndarray): The original board image (BGR format).
        card_detection_model: YOLO model for card detection.

    Returns:
        tuple: (rotated_image, was_rotated) where rotated_image is either the original
               image or rotated 90° clockwise, and was_rotated is a boolean flag.
    """
    # Run card detection on the board image.
    card_results = card_detection_model(board_image)
    # Extract bounding boxes (assuming detections are in the first result).
    card_boxes = card_results[0].boxes.xyxy.cpu().numpy().astype(int)

    # If no cards are detected, return the original image.
    if len(card_boxes) == 0:
        return board_image, False

    total_width = 0
    total_height = 0
    count = 0

    # Calculate total width and height from detected card boxes.
    for box in card_boxes:
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        total_width += width
        total_height += height
        count += 1

    avg_width = total_width / count
    avg_height = total_height / count

    # If cards are generally vertical (taller than wide), rotate the image.
    if avg_height > avg_width:
        rotated_image = cv2.rotate(board_image, cv2.ROTATE_90_CLOCKWISE)
        return rotated_image, True
    else:
        return board_image, False

def detect_cards_from_image(board_image, card_detection_model):
    """
    Detect card regions on the board image using the YOLO card detection model.

    Args:
        board_image (numpy.ndarray): The (corrected) board image.
        card_detection_model: YOLO model for card detection.

    Returns:
        list: List of tuples containing (card_image, bounding_box).
    """
    card_results = card_detection_model(board_image)
    card_boxes = card_results[0].boxes.xyxy.cpu().numpy().astype(int)
    cards = []
    for box in card_boxes:
        x1, y1, x2, y2 = box
        card_img = board_image[y1:y2, x1:x2]
        cards.append((card_img, box))
    return cards
