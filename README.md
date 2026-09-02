# Drowsy Driver Detection System

> This is a fork of [varshathennarasu/DrowsyDriverDetection](https://github.com/varshathennarasu/DrowsyDriverDetection).
> It was a 5-person team project built through the AI Student Collective at UC Davis
> during the Winter 2025 cycle, where it won Best Execution. I worked on it as part
> of that team.

## Overview

This is a real-time system that detects signs of driver drowsiness from a live
webcam feed to help prevent fatigue-related accidents. It analyzes facial landmarks
frame by frame using eye aspect ratio (EAR) thresholds and can play an audio alert
when eyes stay closed. A separate Flask demo can classify uploaded eye images when
a local Keras model file is present.

Kaggle Dataset (optional, for the image scripts): https://www.kaggle.com/datasets/prasadvpatil/mrl-dataset

## Entry points

- **`SourceCode.py`** — Real-time EAR (eye aspect ratio) detector. Uses a webcam, dlib facial landmarks, and EAR thresholds; plays an alert when eyes stay closed. This is the main live demo and does **not** need `drowsiness_model.h5`.
- **`app.py`** — Separate Flask image-upload demo. Serves `index.html`, accepts an uploaded eye image, and classifies it with a Keras model file named `drowsiness_model.h5`. That model file is **not** shipped in this repo; without it the app starts but classification returns a clear error and the UI explains that the demo needs the model.
- **`dataset_processing.py`** / **`test_data.py`** — Optional helpers for exploring or scoring images from a local `kaggle_data/` tree. They are not required for the live EAR demo.

## Features
- **Real-Time Eye Blink Detection**: Monitors eye aspect ratio (EAR) to detect signs of drowsiness.
- **Facial Landmark Detection**: Tracks eye landmarks with dlib’s 68-point predictor.
- **Optional image classification demo**: Flask upload UI when `drowsiness_model.h5` is available locally.
- **Alert Mechanism**: Plays a warning sound (`music.wav`) when prolonged eye closure is detected.
- **Efficient Processing**: Uses `imutils`, `scipy`, and `OpenCV` for the live pipeline.

## Technologies Used
- **Python**: Main programming language.
- **OpenCV**: Video capture and image I/O.
- **dlib**: Facial landmark detection.
- **Pygame**: Audio alerts.
- **scipy** / **imutils**: EAR geometry and frame helpers.
- **Flask** / **TensorFlow** / **Pillow**: Optional upload demo and model inference.

## Project Structure
```
.
├── SourceCode.py                 # Live webcam EAR detector (main demo)
├── app.py                        # Flask image-upload demo
├── index.html                    # Upload UI for app.py
├── test_data.py                  # Optional model smoke test on sample images
├── dataset_processing.py         # Optional Kaggle eye-image explorer
├── music.wav                     # Alert sound
├── requirements.txt              # Python dependencies
├── shape_predictor_68_face_landmarks.dat   # Download separately (see below)
├── drowsiness_model.h5           # Optional; not in repo
└── kaggle_data/                  # Optional; download from Kaggle if needed
```

## How It Works
1. **Eye Aspect Ratio (EAR)**: `SourceCode.py` computes EAR from eye landmarks. If EAR stays below a threshold for enough frames, the driver is treated as drowsy.
2. **Real-Time Alerts**: When the threshold is breached, the system overlays a warning and plays `music.wav`.
3. **Optional upload demo**: `app.py` preprocesses an uploaded grayscale eye crop and runs a local Keras model if present.

## Getting Started
### Prerequisites
```bash
pip install -r requirements.txt
```

Note: `dlib` often needs a working C++ toolchain. On macOS, Xcode Command Line Tools help; on Linux, install build essentials and cmake if the pip wheel is unavailable.

### Live webcam demo (`SourceCode.py`)
1. Download the [shape predictor](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2), decompress it, and place `shape_predictor_68_face_landmarks.dat` in the project root.
2. Confirm `music.wav` is present (shipped in this repo).
3. Run:
   ```bash
   python SourceCode.py
   ```
4. Press `q` to quit. Adjust `thresh` (default `0.25`) and `frame_check` (default `20`) in `SourceCode.py` if needed.

### Flask upload demo (`app.py`)
1. Optionally place a trained `drowsiness_model.h5` in the project root (not included here; no training script is in this fork).
2. Run:
   ```bash
   python app.py
   ```
3. Open the local URL Flask prints and upload an eye image. Without the model file, the page explains what is missing and `/classify` returns a clear JSON error.

### Optional dataset scripts
Download the [MRL eye dataset](https://www.kaggle.com/datasets/prasadvpatil/mrl-dataset) (or equivalent) into `./kaggle_data/train/Closed_Eyes` and `./kaggle_data/train/Open_Eyes`, then run `dataset_processing.py` or `test_data.py`.

## Acknowledgments
This project uses:
- The `shape_predictor_68_face_landmarks` model from [dlib](http://dlib.net/).
- A labeled dataset from [Kaggle](https://www.kaggle.com/datasets/prasadvpatil/mrl-dataset) for optional image exploration / classification.
