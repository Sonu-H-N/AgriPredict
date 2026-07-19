"""
evaluate.py
-----------
Evaluates a trained model on a labeled test directory and prints a
classification report + confusion matrix.

Usage:
    python src/evaluate.py --model models/best_model.h5 --data_dir data/test
"""

import argparse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix


def main():
    parser = argparse.ArgumentParser(description="Evaluate plant disease classifier on a test set.")
    parser.add_argument("--model", required=True, help="Path to trained .h5 model.")
    parser.add_argument("--data_dir", required=True, help="Path to test data directory (class subfolders).")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    model = load_model(args.model)

    datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_gen = datagen.flow_from_directory(
        args.data_dir,
        target_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    print("Running predictions on test set...")
    preds = model.predict(test_gen)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_gen.classes
    class_names = list(test_gen.class_indices.keys())

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names))

    print("Confusion Matrix:\n")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()
