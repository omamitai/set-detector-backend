# SET Detective: Automated Game Solver

## Project Overview

This is a personal project that showcases a full-stack solution for automatically solving the card game "Set". The backend is a Python-based computer vision pipeline that takes an image of a game board, identifies the cards, classifies their features, and detects all valid "sets" according to the game's rules.

The final output is an annotated version of the input image with bounding boxes drawn around the cards that form a set. This project is deployed directly from a GitHub repository.

## Live Demo

You can test the live application with your own images here:
**[SET Detector Live](https://set-detector.lovable.app/)**

---

## How It Works

The core of this project is a computer vision pipeline built with Python, OpenCV, and machine learning models. The models and their weights are located in the `/models` directory.

The identification process follows these steps:
1.  **Image Preprocessing**: The input image is checked for its orientation (portrait vs. landscape) and rotated if necessary to ensure the cards are positioned correctly for detection.
2.  **Card Detection**: A YOLOv8 model detects the location of every card on the board.
3.  **Feature Classification**: For each detected card, a series of models predict its four key features:
    * **Shape** (`diamond`, `oval`, `squiggle`)
    * **Color** (`red`, `green`, `purple`)
    * **Fill** (`empty`, `full`, `striped`)
    * **Count** (`1`, `2`, `3`)
4.  **Set Identification**: The algorithm analyzes all possible combinations of three cards on the board and uses the game's rules to determine which combinations form a valid set. A "set" is valid if, for each of the four features, the attributes are either all the same or all different.
5.  **Output Generation**: A new image is generated, drawing colored bounding boxes around the cards that form valid sets, with each set marked for clarity.

---

## Running Locally

The backend service is built with FastAPI. The models are loaded as YOLO instances, requiring both the model weights (`.pt`) and the corresponding data configuration (`.yaml`) files.

If you are interested in running this project on your local machine, you are welcome to contact me. I would be happy to provide assistance and guidance.

## Contact

For any questions or assistance with a local setup, please feel free to reach out!

* **Omer Amitai**
* **Email**: [omermamitai@gmail.com](mailto:omermamitai@gmail.com)
