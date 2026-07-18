"""
dataset.py
----------
Data loading, augmentation, and optional train/val/test splitting utilities
for the Plant Disease Prediction project.
"""

import os
import shutil
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def get_data_generators(data_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """
    Creates train/val/test Keras ImageDataGenerators from a directory
    structured as:
        data_dir/train/<class_name>/*.jpg
        data_dir/val/<class_name>/*.jpg
        data_dir/test/<class_name>/*.jpg   (optional)

    Returns: (train_generator, val_generator, test_generator_or_None)
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode="nearest",
    )

    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )

    val_gen = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    test_gen = None
    if os.path.isdir(test_dir):
        test_gen = val_test_datagen.flow_from_directory(
            test_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=False,
        )

    return train_gen, val_gen, test_gen


def split_dataset(source_dir, dest_dir, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Splits a single folder of class subfolders into train/val/test folders.

    source_dir/
        ClassA/*.jpg
        ClassB/*.jpg
    ->
    dest_dir/train/ClassA/*.jpg, dest_dir/val/ClassA/*.jpg, dest_dir/test/ClassA/*.jpg
    (and same for ClassB, etc.)

    test_ratio is implied as 1 - train_ratio - val_ratio.
    """
    random.seed(seed)
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    for cls in classes:
        cls_path = os.path.join(source_dir, cls)
        images = [f for f in os.listdir(cls_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        for split_name, files in splits.items():
            split_class_dir = os.path.join(dest_dir, split_name, cls)
            os.makedirs(split_class_dir, exist_ok=True)
            for f in files:
                shutil.copy2(os.path.join(cls_path, f), os.path.join(split_class_dir, f))

        print(f"[{cls}] total={n_total} train={len(splits['train'])} "
              f"val={len(splits['val'])} test={len(splits['test'])}")

    print(f"\nDone. Split dataset written to: {dest_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Split a flat class-folder dataset into train/val/test.")
    parser.add_argument("--source_dir", required=True, help="Folder containing one subfolder per class.")
    parser.add_argument("--dest_dir", required=True, help="Output folder for split dataset.")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    args = parser.parse_args()

    split_dataset(args.source_dir, args.dest_dir, args.train_ratio, args.val_ratio)
