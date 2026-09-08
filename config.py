"""
CONFIG — the two things you MUST edit to match your model
============================================================
1. IMG_SIZE  -> the input size your model was trained on (e.g. 224x224, 128x128)
2. CLASS_NAMES -> the list of class names IN THE EXACT ORDER your model outputs them.

If you trained with Keras' ImageDataGenerator / image_dataset_from_directory,
the order is alphabetical by folder name. You can get it back with:

    import json
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    gen = ImageDataGenerator().flow_from_directory("path/to/train/data")
    print(gen.class_indices)   # {'Apple___Black_rot': 0, 'Apple___healthy': 1, ...}

Copy that order into CLASS_NAMES below. If the order is wrong, every prediction
will point to the wrong disease name (the confidence numbers will still look
fine, so this bug is silent — double check it).
"""

MODEL_PATH = "\model\plant_disease_efficientnetb0.keras"

IMG_SIZE = (224, 224)


CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

# Below this confidence, the API tells the user "not sure" instead of guessing
CONFIDENCE_THRESHOLD = 0.55
