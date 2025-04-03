import cv2
import numpy as np
from collections import defaultdict

# Extended color palette with 20 distinct colors for better set representation
# BGR format for OpenCV compatibility
COLORS = [
    (255, 0, 0),     # Blue
    (0, 255, 0),     # Green
    (0, 0, 255),     # Red
    (255, 255, 0),   # Cyan
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Yellow
    (128, 0, 0),     # Dark blue
    (0, 128, 0),     # Dark green
    (0, 0, 128),     # Dark red
    (128, 128, 0),   # Teal
    (128, 0, 128),   # Purple
    (0, 128, 128),   # Brown
    (255, 128, 0),   # Light blue
    (255, 0, 128),   # Pink
    (128, 255, 0),   # Light green
    (0, 255, 128),   # Mint
    (128, 0, 255),   # Violet
    (0, 128, 255),   # Orange
    (255, 128, 128), # Light cyan
    (128, 255, 128)  # Light yellow
]

# Cache for parsed coordinates to avoid repeated parsing
_coord_cache = {}

def parse_coordinates(coord_str):
    """
    Parse coordinate string to integer values with caching for performance.
    
    Args:
        coord_str (str): Comma-separated coordinate string "x1, y1, x2, y2"
    
    Returns:
        tuple: (x1, y1, x2, y2) as integers
    """
    # Check if already in cache
    if coord_str in _coord_cache:
        return _coord_cache[coord_str]
    
    # Parse coordinates and store in cache
    coords = tuple(map(int, coord_str.split(',')))
    _coord_cache[coord_str] = coords
    return coords

def draw_sets_on_image(board_image, sets_info):
    """
    Draw bounding boxes and labels for the detected sets on the provided board image.
    
    Args:
        board_image (numpy.ndarray): Board image in BGR format.
        sets_info (list): List of dictionaries containing set information.
        
    Returns:
        numpy.ndarray: The annotated board image.
    """
    # Early return for empty input
    if not sets_info:
        return board_image.copy()
    
    # Clear coordinate cache for new image
    _coord_cache.clear()
        
    # Get image dimensions and calculate scaling factors once
    img_height, img_width = board_image.shape[:2]
    img_diagonal = np.sqrt(img_width**2 + img_height**2)
    
    # Scale parameters based on image size - calculate once
    base_thickness = max(1, int(img_diagonal * 0.004))
    base_expansion = max(5, int(img_diagonal * 0.008))
    font_scale = max(0.5, img_diagonal * 0.0007)
    text_margin = max(5, int(img_diagonal * 0.008))
    
    # Create a copy of the image - only do this once
    result_image = board_image.copy()
    
    # Pre-process all card data to avoid repeated calculations
    card_data = {}
    
    # First pass: Efficiently gather all card data and set membership
    for set_idx, set_info in enumerate(sets_info):
        for card in set_info['cards']:
            # Get coordinates
            if isinstance(card['Coordinates'], str):
                coord_str = card['Coordinates']
                coords = parse_coordinates(coord_str)
            else:
                coords = tuple(card['Coordinates'])
                coord_str = f"{coords[0]}, {coords[1]}, {coords[2]}, {coords[3]}"
            
            # Initialize card data if not already seen
            if coord_str not in card_data:
                card_data[coord_str] = {
                    'coords': coords,
                    'sets': []
                }
            
            # Add set membership
            card_data[coord_str]['sets'].append(set_idx)
    
    # Prepare all rectangles to draw in a batch for better performance
    rectangles_to_draw = []
    labels_to_draw = []
    
    # Generate all rectangles and labels
    for coord_str, data in card_data.items():
        x1, y1, x2, y2 = data['coords']
        set_indices = data['sets']
        
        # Calculate appropriate expansions based on total appearances
        total_appearances = len(set_indices)
        expansions = []
        
        if total_appearances == 1:
            expansions = [0]
        elif total_appearances == 2:
            expansions = [0, base_expansion]
        else:
            expansions = [0, base_expansion]
            # Add 2x base_expansion for 3rd+ appearances
            expansions.extend([2 * base_expansion] * (total_appearances - 2))
        
        # Generate rectangles for each appearance
        for i, set_idx in enumerate(set_indices):
            expansion = expansions[i]
            color = COLORS[set_idx % len(COLORS)]
            
            # Calculate expanded coordinates with bounds checking
            x1_exp = max(0, x1 - expansion)
            y1_exp = max(0, y1 - expansion)
            x2_exp = min(img_width, x2 + expansion)
            y2_exp = min(img_height, y2 + expansion)
            
            # Store rectangle data for batch drawing
            rectangles_to_draw.append({
                'coords': (x1_exp, y1_exp, x2_exp, y2_exp),
                'color': color,
                'thickness': base_thickness,
                'set_idx': set_idx,
                'is_first_card': False  # Will update below
            })
    
    # Mark first card in each set (for labels)
    for set_idx, set_info in enumerate(sets_info):
        if not set_info['cards']:
            continue
            
        first_card = set_info['cards'][0]
        
        # Get coordinates
        if isinstance(first_card['Coordinates'], str):
            coord_str = first_card['Coordinates']
        else:
            coords = tuple(first_card['Coordinates'])
            coord_str = f"{coords[0]}, {coords[1]}, {coords[2]}, {coords[3]}"
        
        # Find the rectangle for this card and set
        for rect in rectangles_to_draw:
            if (rect['set_idx'] == set_idx and 
                coord_str in card_data and 
                set_idx in card_data[coord_str]['sets']):
                
                rect['is_first_card'] = True
                
                # Prepare label
                x1_exp, y1_exp, _, _ = rect['coords']
                text_y = max(text_margin, y1_exp - text_margin)
                
                # Store label data for batch drawing
                labels_to_draw.append({
                    'text': f"Set {set_idx + 1}",
                    'position': (x1_exp, text_y),
                    'font_face': cv2.FONT_HERSHEY_SIMPLEX,
                    'font_scale': font_scale,
                    'color': rect['color'],
                    'thickness': base_thickness
                })
                break
    
    # Draw all rectangles at once
    for rect in rectangles_to_draw:
        cv2.rectangle(
            result_image,
            (rect['coords'][0], rect['coords'][1]),
            (rect['coords'][2], rect['coords'][3]),
            rect['color'],
            rect['thickness']
        )
    
    # Draw all labels at once
    for label in labels_to_draw:
        cv2.putText(
            result_image,
            label['text'],
            label['position'],
            label['font_face'],
            label['font_scale'],
            label['color'],
            label['thickness']
        )
    
    return result_image
