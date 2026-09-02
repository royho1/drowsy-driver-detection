from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from pygame import mixer
from PIL import Image
import numpy as np
import os

# index.html lives next to this file (not under templates/)
app = Flask(__name__, template_folder=".")

MODEL_PATH = "drowsiness_model.h5"
_model = None

# Initialize pygame mixer for sound
mixer.init()
if os.path.exists("music.wav"):
    mixer.music.load("music.wav")


def get_model():
    """Load the Keras model on first use. Returns (model, error_message)."""
    global _model
    if _model is not None:
        return _model, None
    if not os.path.exists(MODEL_PATH):
        return None, (
            f"Model file '{MODEL_PATH}' is missing. "
            "Place a trained Keras .h5 model in the project root to use this demo."
        )
    try:
        _model = load_model(MODEL_PATH)
        print("Model loaded successfully.")
        return _model, None
    except Exception as e:
        return None, f"Failed to load model '{MODEL_PATH}': {e}"


def preprocess_image(image):
    img = image.resize((64, 64))  # Resize to match model input size
    img = img.convert("L")  # Convert to grayscale
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = img_array.reshape(1, 64, 64, 1)  # Reshape to match model input shape
    return img_array


@app.route("/")
def index():
    model_available = os.path.exists(MODEL_PATH)
    return render_template("index.html", model_available=model_available)


@app.route("/classify", methods=["POST"])
def classify_image():
    model, model_error = get_model()
    if model is None:
        return jsonify({"error": model_error}), 503

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    try:
        if file:
            image = Image.open(file)
            processed_image = preprocess_image(image)
            print("Image preprocessed successfully.")

            prediction = model.predict(processed_image)
            print("Prediction result:", prediction)

            # Drowsy if Closed Eyes class is more likely
            if prediction[0][0] > prediction[0][1]:
                result = "Drowsy Driver: Closed Eyes"
                if mixer.get_init() and os.path.exists("music.wav"):
                    mixer.music.play()
            else:
                result = "Alert Driver: Open Eyes"

            return jsonify({"prediction": result})

        return jsonify({"error": "No file uploaded"}), 400

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify({"error": f"Error processing image: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
