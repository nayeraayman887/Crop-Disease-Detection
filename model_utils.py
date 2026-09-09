import io
import numpy as np
from PIL import Image
import tensorflow as tf

import config

_model = None  # loaded once, reused for every request


def load_model():
    """Loads the .h5 model into memory. Called once when the server starts."""
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(config.MODEL_PATH)
    return _model


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Takes raw uploaded image bytes and turns them into the array shape
    your model expects: (1, height, width, 3).

    This uses EfficientNetB0's own preprocessing function, NOT a plain
    /255.0 scale. EfficientNet was trained with a specific scaling/offset
    baked into tf.keras.applications.efficientnet.preprocess_input, and
    using plain /255.0 instead will silently produce wrong predictions
    (no error, just bad confidence scores).

    If it turns out your teammate actually trained EfficientNetB0 WITH a
    Rescaling(1./255) layer built into the model itself (common in some
    tutorials), switch back to plain /255.0 here instead - ask whoever
    built the model which one they used.
    """
    from tensorflow.keras.applications.efficientnet import preprocess_input

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(config.IMG_SIZE)
    array = np.array(image, dtype=np.float32)
    array = np.expand_dims(array, axis=0)
    array = preprocess_input(array)
    return array


def predict(image_bytes: bytes):
    """Runs the model on an image and returns (class_name, probability)."""

    model = load_model()
    array = preprocess_image(image_bytes)

    predictions = model.predict(array, verbose=0)[0]

    probabilities = tf.nn.softmax(predictions).numpy()

    top_index = int(np.argmax(probabilities))

    confidence = float(probabilities[top_index])

    if top_index >= len(config.CLASS_NAMES):
        raise ValueError(
            "Model output has more classes than CLASS_NAMES in config.py. "
            "Update CLASS_NAMES to match your model's training classes."
        )

    class_name = config.CLASS_NAMES[top_index]

    return class_name, confidence
