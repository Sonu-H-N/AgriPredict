"""
app.py
------
Simple Flask web app: upload a leaf photo, get an instant disease
prediction from the trained model.

Usage:
    python app.py
Then open http://localhost:5000
"""

import os
import sys
import json
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from disease_info import get_disease_info

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_ROOT, "models", "best_model.h5")
CLASS_MAP_PATH = os.path.join(APP_ROOT, "models", "class_indices.json")
UPLOAD_DIR = os.path.join(APP_ROOT, "static", "uploads")
IMG_SIZE = (224, 224)
ALLOWED_EXT = {"png", "jpg", "jpeg"}

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR

_model = None
_idx_to_class = None


def get_model():
    """Lazy-load the model so the app can still start even before training."""
    global _model, _idx_to_class
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"No trained model found at {MODEL_PATH}. Run src/train.py first."
            )
        _model = load_model(MODEL_PATH)
        with open(CLASS_MAP_PATH, "r") as f:
            raw_map = json.load(f)
        _idx_to_class = {int(k): v for k, v in raw_map.items()}
    return _model, _idx_to_class


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def predict(image_path, top_k=3):
    model, idx_to_class = get_model()
    img = image.load_img(image_path, target_size=IMG_SIZE)
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr)[0]
    top_indices = preds.argsort()[-top_k:][::-1]
    return [(idx_to_class[i], float(preds[i]) * 100) for i in top_indices]


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    if "file" not in request.files:
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return render_template("index.html", error="Please upload a valid image file (png/jpg/jpeg).")

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    try:
        results = predict(save_path)
    except FileNotFoundError as e:
        return render_template("index.html", error=str(e))

    top_label = results[0][0]
    top_info = get_disease_info(top_label)

    image_url = url_for("static", filename=f"uploads/{filename}")
    return render_template(
        "index.html",
        results=results,
        image_url=image_url,
        top_info=top_info,
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
