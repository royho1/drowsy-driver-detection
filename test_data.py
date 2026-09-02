from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
import sys

MODEL_PATH = "drowsiness_model.h5"


def load_drowsiness_model():
    if not os.path.exists(MODEL_PATH):
        print(
            f"Error: '{MODEL_PATH}' not found. "
            "Place a trained Keras .h5 model in the project root before running this script."
        )
        return None
    try:
        return load_model(MODEL_PATH)
    except Exception as e:
        print(f"Error: failed to load '{MODEL_PATH}': {e}")
        return None


def test_model_on_image(model, image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not read image: {image_path}")
        return
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = img.reshape(1, 64, 64, 1)

    prediction = model.predict(img)

    if prediction[0][0] > prediction[0][1]:
        print("Prediction: Closed Eyes")
    else:
        print("Prediction: Open Eyes")


if __name__ == "__main__":
    model = load_drowsiness_model()
    if model is None:
        sys.exit(1)

    test_model_on_image(model, "./kaggle_data/train/Closed_Eyes/s0001_00001_0_0_0_0_0_01.png")
    test_model_on_image(model, "./kaggle_data/train/Open_Eyes/s0001_02334_0_0_1_0_0_01.png")
