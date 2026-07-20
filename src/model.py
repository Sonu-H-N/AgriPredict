"""
model.py
--------
CNN and transfer-learning (MobileNetV2) model architectures for
plant disease image classification.
"""

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2


def build_cnn(input_shape=(224, 224, 3), num_classes=10):
    """A custom CNN built from scratch — good baseline, trains fast."""
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    return model


def build_mobilenet(input_shape=(224, 224, 3), num_classes=10, fine_tune=False):
    """
    Transfer-learning model using MobileNetV2 pretrained on ImageNet.
    Recommended for higher accuracy, especially with limited training data.
    """
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = fine_tune  # freeze by default

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    return model


def get_model(model_type, input_shape, num_classes):
    if model_type == "mobilenet":
        return build_mobilenet(input_shape, num_classes)
    return build_cnn(input_shape, num_classes)
