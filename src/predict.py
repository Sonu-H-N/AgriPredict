"""
predict.py
----------
Loads a trained model and predicts the disease class for a single
leaf image.

Usage:
    python src/predict.py --model models/best_model.h5 --image path/to/leaf.jpg
"""

import os
import json
import argparse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


def load_class_map(model_dir):
    class_map_path = os.path.join(model_dir, "class_indices.json")
    with open(class_map_path, "r") as f:
        idx_to_class = json.load(f)
    # JSON keys are strings; convert back to int
    return {int(k): v for k, v in idx_to_class.items()}


def predict_image(model_path, image_path, img_size=(224, 224), top_k=3):
    model_dir = os.path.dirname(model_path) or "."
    idx_to_class = load_class_map(model_dir)

    model = load_model(model_path)

    img = image.load_img(image_path, target_size=img_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]
    top_indices = preds.argsort()[-top_k:][::-1]

    results = [(idx_to_class[i], float(preds[i]) * 100) for i in top_indices]
    return results


def main():
    parser = argparse.ArgumentParser(description="Predict plant disease from a leaf image.")
    parser.add_argument("--model", required=True, help="Path to trained .h5 model.")
    parser.add_argument("--image", required=True, help="Path to the leaf image.")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    results = predict_image(
        args.model, args.image, img_size=(args.img_size, args.img_size), top_k=args.top_k
    )

    print(f"\nPredictions for: {args.image}\n" + "-" * 40)
    for rank, (label, confidence) in enumerate(results, start=1):
        marker = "-->" if rank == 1 else "   "
        print(f"{marker} {rank}. {label:<35} {confidence:6.2f}%")


if __name__ == "__main__":
    main()
