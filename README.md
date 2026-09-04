# Drowsy Driver Detection System

## Overview

A real-time computer vision system that detects signs of driver drowsiness from a live webcam feed to help prevent fatigue-related accidents. The system analyzes facial landmarks frame by frame using eye aspect ratio (EAR) thresholds and triggers an audio alert when the driver's eyes remain closed for an extended period. A separate Flask demo can also classify uploaded eye images using a local Keras model. This was a five-person team project developed through the AI Student Collective at UC Davis during the Winter 2025 cycle, where it won **Best Execution**.


Kaggle Dataset (optional, for the image scripts): https://www.kaggle.com/datasets/prasadvpatil/mrl-dataset

## Interview / live demo (resume-accurate path)

This is the path that matches the eye-state / EAR webcam work.

**Run from Terminal.app** (macOS Camera permission often fails when launched from an IDE):

```bash
cd "/path/to/drowsy-driver-detection"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python setup_live_demo.py          # downloads dlib's 68-point landmark model
python SourceCode.py               # webcam window; press q to quit
```

If the camera fails: **System Settings → Privacy & Security → Camera** → enable Terminal.

Close your eyes for ~70 frames (a few seconds) to trigger **DROWSINESS ALERT!** and `music.wav`.
Defaults: EAR threshold `0.25`, consecutive-frame check `70`.

## Entry points

- **`SourceCode.py`** — Real-time EAR (eye aspect ratio) detector. Uses a webcam, dlib facial landmarks, and EAR thresholds; plays an alert when eyes stay closed. This is the main live demo and does **not** need `drowsiness_model.h5`.
- **`setup_live_demo.py`** — Downloads and decompresses `shape_predictor_68_face_landmarks.dat` into the project root.
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
- **OpenCV**: Video capture, grayscale preprocessing, and convex-hull eye overlays.
- **dlib**: Frontal face detector and 68-point facial landmark model.
- **Pygame**: Audio alerts.
- **scipy** / **imutils**: EAR Euclidean distances and frame helpers.
- **Flask** / **TensorFlow** / **Pillow**: Optional upload demo and model inference (`requirements-optional.txt`).

## Project Structure
```
.
├── SourceCode.py                 # Live webcam EAR detector (main demo)
├── setup_live_demo.py            # Downloads the dlib shape predictor
├── app.py                        # Flask image-upload demo
├── index.html                    # Upload UI for app.py
├── test_data.py                  # Optional model smoke test on sample images
├── dataset_processing.py         # Optional Kaggle eye-image explorer
├── music.wav                     # Alert sound
├── requirements.txt              # Live EAR demo dependencies
├── requirements-optional.txt     # Flask / TensorFlow extras
├── shape_predictor_68_face_landmarks.dat   # Via setup_live_demo.py (gitignored)
├── drowsiness_model.h5           # Optional; not in repo
└── kaggle_data/                  # Optional; download from Kaggle if needed
```

## How It Works
1. **Eye Aspect Ratio (EAR)**: `SourceCode.py` computes EAR from eye landmarks via scipy Euclidean distance. If EAR stays below `0.25` for `70` consecutive frames, the driver is treated as drowsy.
2. **Real-Time Alerts**: When the threshold is breached, the system overlays a warning, draws convex-hull eye contours, and plays `music.wav`.
3. **Missing assets**: If the landmark model is missing, the script exits with setup instructions (run `setup_live_demo.py`). Missing `music.wav` keeps visual alerts working.
4. **Optional upload demo**: `app.py` preprocesses an uploaded grayscale eye crop and runs a local Keras model if present.

## Getting Started
### Prerequisites
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python setup_live_demo.py
```

Note: `dlib` often needs a working C++ toolchain. On macOS, Xcode Command Line Tools help; on Linux, install build essentials and cmake if the pip wheel is unavailable.

### Live webcam demo (`SourceCode.py`)
1. Confirm `music.wav` is present (shipped in this repo) and the shape predictor was downloaded.
2. Run:
   ```bash
   python SourceCode.py
   ```
3. Press `q` to quit. Adjust `thresh` (default `0.25`) and `frame_check` (default `70`) in `SourceCode.py` if needed.

### Flask upload demo (`app.py`)
1. Optionally place a trained `drowsiness_model.h5` in the project root (not included here; no training script is in this fork).
2. Install extras: `pip install -r requirements-optional.txt`
3. Run:
   ```bash
   python app.py
   ```
4. Open the local URL Flask prints and upload an eye image. Without the model file, the page explains what is missing and `/classify` returns a clear JSON error.

### Optional dataset scripts
Download the [MRL eye dataset](https://www.kaggle.com/datasets/prasadvpatil/mrl-dataset) (or equivalent) into `./kaggle_data/train/Closed_Eyes` and `./kaggle_data/train/Open_Eyes`, then run `dataset_processing.py` or `test_data.py`.

## Acknowledgments
This project uses:
- The `shape_predictor_68_face_landmarks` model from [dlib](http://dlib.net/).
- A labeled dataset from [Kaggle](https://www.kaggle.com/datasets/prasadvpatil/mrl-dataset) for optional image exploration / classification.
