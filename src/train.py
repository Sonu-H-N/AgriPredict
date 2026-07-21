"""
train.py
--------
Trains the rice leaf disease classification model and saves the best
checkpoint, the class-index mapping, and a training-history plot.

Usage:
    python src/train.py --data_dir data --epochs 25 --batch_size 32 --model_type cnn

Tip: use --model_type mobilenet for better results on small rice datasets
(transfer learning generalizes better with limited training images).
"""

import os
import json
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from dataset import get_data_generators
from model import get_model


def plot_history(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="train_acc")
    axes[0].plot(history.history["val_accuracy"], label="val_acc")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train_loss")
    axes[1].plot(history.history["val_loss"], label="val_loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved training curves to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Train plant disease classifier.")
    parser.add_argument("--data_dir", default="data", help="Root data dir with train/val/test subfolders.")
    parser.add_argument("--model_dir", default="models", help="Where to save trained model artifacts.")
    parser.add_argument("--model_type", choices=["cnn", "mobilenet"], default="cnn")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    img_size = (args.img_size, args.img_size)

    print("Loading data...")
    train_gen, val_gen, test_gen = get_data_generators(
        args.data_dir, img_size=img_size, batch_size=args.batch_size
    )

    num_classes = len(train_gen.class_indices)
    print(f"Found {num_classes} classes: {list(train_gen.class_indices.keys())}")

    # Save class index mapping (index -> class name), needed at inference time
    idx_to_class = {v: k for k, v in train_gen.class_indices.items()}
    class_map_path = os.path.join(args.model_dir, "class_indices.json")
    with open(class_map_path, "w") as f:
        json.dump(idx_to_class, f, indent=2)
    print(f"Saved class index map to {class_map_path}")

    print(f"Building '{args.model_type}' model...")
    model = get_model(args.model_type, input_shape=(*img_size, 3), num_classes=num_classes)
    model.compile(
        optimizer=Adam(learning_rate=args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    checkpoint_path = os.path.join(args.model_dir, "best_model.h5")
    callbacks = [
        ModelCheckpoint(checkpoint_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ]

    print("Starting training...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    plot_history(history, os.path.join(args.model_dir, "training_history.png"))

    if test_gen is not None:
        print("Evaluating on test set...")
        test_loss, test_acc = model.evaluate(test_gen)
        print(f"Test accuracy: {test_acc * 100:.2f}%  |  Test loss: {test_loss:.4f}")

    print(f"\nBest model saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
