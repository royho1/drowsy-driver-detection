# Drowsy Driver Detection System

> This is a fork of [varshathennarasu/DrowsyDriverDetection](https://github.com/varshathennarasu/DrowsyDriverDetection).
> It was a 5-person team project built through the AI Student Collective at UC Davis
> during the Winter 2025 cycle, where it won Best Execution. I worked on it as part
> of that team.

## Overview

This is a real-time system that detects signs of driver drowsiness from a live
webcam feed to help prevent fatigue-related accidents. It analyzes facial landmarks
frame by frame and pairs threshold-based eye tracking with a machine learning model
trained on a labeled Kaggle dataset, then triggers an audio alert when it detects
fatigue.

Kaggle Dataset: https://www.kaggle.com/datasets/prasadvpatil/mrl-dataset

## Features
- **Real-Time Eye Blink Detection**: Monitors eye aspect ratio (EAR) to detect signs of drowsiness.
- **Facial Landmark Detection**: Identifies facial regions to track eye and mouth movements using a pre-trained model.
- **Machine Learning Integration**: Trains and evaluates a model using a Kaggle dataset for more accurate drowsiness and intoxication detection.
- **Alert Mechanism**: Plays a warning sound if drowsiness or intoxication is detected.
- **Efficient Processing**: Optimized for quick response times using `imutils`, `scipy`, and `OpenCV`.

## Technologies Used
- **Python**: Main programming language.
- **OpenCV**: For video capture and facial detection.
- **dlib**: For detecting facial landmarks.
- **Pygame**: For audio alerts.
- **scipy**: To compute Euclidean distances for EAR.
- **Pandas & NumPy**: For data manipulation during model training.
- **Scikit-learn**: For training and evaluating the machine learning model.

## Kaggle Dataset
This project utilizes a dataset from [Kaggle](https://www.kaggle.com/) containing labeled images of drowsy and alert drivers. The dataset was preprocessed and split into training and testing sets to develop a robust machine learning model capable of detecting drowsy behavior.

## Project Structure
```
.
├── data/                                       # Dataset folder
│   ├── train/                                 # Training data
│   └── test/                                  # Testing data
├── shape_predictor_68_face_landmarks.dat      # Pre-trained model for facial landmarks
├── music.wav                                  # Alert sound file
├── drunk_drowsy_detection.py                  # Main script
└── README.md                                  # Project documentation
```

## How It Works
1. **Eye Aspect Ratio (EAR)**: Calculates EAR from facial landmarks. If EAR falls below a threshold, the driver is likely drowsy.
2. **Facial Landmark Analysis**: Tracks mouth movements to detect potential yawning (a sign of drowsiness).
3. **Machine Learning Model**: Analyzes features extracted from facial landmarks and predicts driver status (alert or drowsy).
4. **Real-Time Alerts**: When thresholds are breached or the model predicts drowsiness, the system triggers an alert.

## Getting Started
### Prerequisites
Ensure you have the following Python libraries installed:
- `OpenCV`
- `dlib`
- `pygame`
- `imutils`
- `scipy`
- `pandas`
- `numpy`
- `scikit-learn`

Install the required packages using:
```bash
pip install opencv-python dlib pygame imutils scipy pandas numpy scikit-learn
```

### Usage
1. Download the [shape predictor model](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and place it in the project directory.
2. Ensure the Kaggle dataset is downloaded and placed in the `data/` directory.
3. Run the script:
   ```bash
   python drunk_drowsy_detection.py
   ```
4. The system will start the camera, monitor driver behavior in real-time, and utilize the trained model for predictions.

### Configuration
- Modify `thresh` (default: `0.25`) in the script to adjust the sensitivity of EAR.
- Adjust `frame_check` (default: `20`) to set the number of consecutive frames required to trigger the alert.

## Future Improvements
- Integrating additional features like head pose estimation for better accuracy.
- Deploying the system on mobile devices or in-vehicle hardware.
- Incorporating deep learning for advanced behavioral analysis.

## Acknowledgments
This project uses:
- The `shape_predictor_68_face_landmarks` model from [dlib](http://dlib.net/).
- A labeled dataset from Kaggle(https://www.kaggle.com/datasets/prasadvpatil/mrl-dataset) for training and testing.

