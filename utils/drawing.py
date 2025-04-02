import cv2

def draw_sets_on_image(board_image, sets_info):
    """
    Draw bounding boxes and labels for the detected sets on the provided board image.

    Args:
        board_image (numpy.ndarray): Board image in BGR format.
        sets_info (list): List of dictionaries containing set information.

    Returns:
        numpy.ndarray: The annotated board image.
    """
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
              (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    base_thickness = 8
    base_expansion = 5
    for index, set_info in enumerate(sets_info):
        color = colors[index % len(colors)]
        thickness = base_thickness + 2 * index
        expansion = base_expansion + 15 * index
        for i, card in enumerate(set_info['cards']):
            coordinates = list(map(int, card['Coordinates'].split(',')))
            x1, y1, x2, y2 = coordinates
            x1_expanded = max(0, x1 - expansion)
            y1_expanded = max(0, y1 - expansion)
            x2_expanded = min(board_image.shape[1], x2 + expansion)
            y2_expanded = min(board_image.shape[0], y2 + expansion)
            cv2.rectangle(board_image, (x1_expanded, y1_expanded), (x2_expanded, y2_expanded), color, thickness)
            if i == 0:
                cv2.putText(board_image, f"Set {index + 1}", (x1_expanded, y1_expanded - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thickness)
    return board_image
