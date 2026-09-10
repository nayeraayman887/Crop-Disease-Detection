import io
import json
import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# MODEL CONFIG

MODEL_PATH = "models/plant_disease_efficientnetb0.keras"
CLASS_NAMES_PATH = "models/class_names.json"
IMG_SIZE = (224, 224)  # confirm this matches the training notebook
CONFIDENCE_THRESHOLD = 0.55

FALLBACK_CLASS_NAMES = [
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

# MODEL LOADING (cached so it only happens once, not per click)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource
def load_class_names():
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [name for name, _ in sorted(data.items(), key=lambda kv: kv[1])]

    return FALLBACK_CLASS_NAMES


@st.cache_resource
def load_disease_info():
    with open("disease_info.json", "r", encoding="utf-8") as f:
        return json.load(f)


def preprocess_image(image: Image.Image) -> np.ndarray:
    # No manual scaling here on purpose: resnet_v2.preprocess_input is
    # built into the model graph itself, so we send raw 0-255 pixel values.
    image = image.convert("RGB").resize(IMG_SIZE)
    array = np.array(image, dtype=np.float32)
    array = np.expand_dims(array, axis=0)
    return array


def analyze_image(file_bytes: bytes, filename: str, content_type: str) -> dict:

    try:
        model = load_model()
        class_names = load_class_names()
        disease_info = load_disease_info()

        image = Image.open(io.BytesIO(file_bytes))
        array = preprocess_image(image)
        predictions = np.asarray(model.predict(array, verbose=0))
        predictions = predictions.reshape(-1)

        if predictions.size != len(class_names):
            return {
                "status": "error",
                "message": "Model output size does not match the number of class "
                           "names. Update class_names.json (or the fallback list) "
                           "to match the model's actual output classes.",
            }

        probabilities = tf.nn.softmax(predictions).numpy()

        top_index = int(np.argmax(probabilities))
        confidence = float(probabilities[top_index])

        if confidence < CONFIDENCE_THRESHOLD:
            return {
                "status": "uncertain",
                "message": "Could not confidently identify a disease. Try a clearer, "
                           "closer photo of the affected leaf.",
                "confidence": confidence,
            }

        class_name = class_names[top_index]
        info = disease_info.get(class_name)

        if info is None:
            return {
                "status": "success",
                "raw_class": class_name,
                "confidence": confidence,
                "message": "Prediction succeeded but no treatment info is on file "
                           "for this class. Add an entry for it in disease_info.json.",
            }

        return {
            "status": "success",
            "disease": info["common_name"],
            "confidence": confidence,
            "description": info["description"],
            "treatment": info["treatment"],
            "prevention": info["prevention"],
        }

    except Exception as e:
        return {"status": "error", "message": f"Unexpected error: {e}"}

# PAGE CONFIGURATION

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)
# CUSTOM CSS

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background-color: #f7faf7;
    }

    /* Main title */
    .main-title {
        font-size: 45px;
        font-weight: 700;
        color: #2e7d32;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #666666;
        font-size: 18px;
        margin-bottom: 30px;
    }

    /* Result cards */
    .result-card {
        padding: 20px;
        border-radius: 15px;
        background-color: white;
        border: 1px solid #e0e0e0;
        margin-bottom: 15px;
    }

    .healthy {
        padding: 20px;
        border-radius: 15px;
        background-color: #e8f5e9;
        border-left: 6px solid #43a047;
        margin-bottom: 20px;
    }

    .diseased {
        padding: 20px;
        border-radius: 15px;
        background-color: #ffebee;
        border-left: 6px solid #e53935;
        margin-bottom: 20px;
    }

    .uncertain {
        padding: 20px;
        border-radius: 15px;
        background-color: #fff8e1;
        border-left: 6px solid #f9a825;
        margin-bottom: 20px;
    }

    .section-title {
        color: #2e7d32;
        font-size: 24px;
        font-weight: 600;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #777;
        margin-top: 50px;
        padding: 20px;
    }

</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown(
    '<div class="main-title">🌿 Plant Disease Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Detect plant diseases using AI and get recommended treatment solutions.'
    '</div>',
    unsafe_allow_html=True
)

# SIDEBAR
with st.sidebar:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/2909/2909761.png",
        width=100
    )

    st.title("🌱 green cure")
    st.write(
        "Upload a photo of a plant leaf or take a picture using your camera "
        "to detect possible diseases."
    )
    st.divider()
    st.subheader("How the app works")

    st.write("1️⃣ Upload or take a plant photo")
    st.write("2️⃣ AI analyzes the image")
    st.write("3️⃣ Disease is detected")
    st.write("4️⃣ Get treatment recommendations")
    st.divider()

# INPUT TABS

tab1, tab2 = st.tabs([
    "📤 Upload Image",
    "📷 Take a Photo"
])

# FUNCTION: DISPLAY RESULT
def display_result(result: dict):

    st.markdown(
        '<div class="section-title">🔍 Detection Result</div>',
        unsafe_allow_html=True
    )

    status = result.get("status")
    # Something went wrong during inference

    if status == "error":
        st.error(result.get("message", "Something went wrong analyzing the image."))
        return

    # Model wasn't confident enough (below CONFIDENCE_THRESHOLD)

    if status == "uncertain":
        confidence = result.get("confidence", 0) * 100
        st.markdown(
            f"""
            <div class="uncertain">
                <h2>🤔 Not Sure</h2>
                <p>{result.get("message", "Could not confidently identify a disease.")}</p>
                <p><b>Confidence:</b> {confidence:.1f}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    if status != "success":
        st.error(result.get("message", "Unexpected result from the model."))
        return

    disease_name = result.get("disease")
    confidence = result.get("confidence", 0) * 100

    # Model predicted a class but disease_info.json has no entry for it
    if disease_name is None:
        st.markdown(
            f"""
            <div class="diseased">
                <h2>⚠️ Prediction Only</h2>
                <p><b>Detected class:</b> {result.get("raw_class", "Unknown")}</p>
                <p><b>Confidence:</b> {confidence:.1f}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.info(result.get("message", ""))
        return
    # STATUS — normal healthy / diseased result
    is_healthy = disease_name.strip().lower() == "healthy"

    if not is_healthy:

        st.markdown(
            f"""
            <div class="diseased">
                <h2>⚠️ Disease Detected</h2>
                <p><b>Disease:</b> {disease_name}</p>
                <p><b>Confidence:</b> {confidence:.1f}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:

        st.markdown(
            f"""
            <div class="healthy">
                <h2>✅ Plant Looks Healthy</h2>
                <p>No significant disease was detected.</p>
                <p><b>Confidence:</b> {confidence:.1f}%</p>
            </div>
            """,
            unsafe_allow_html=True)

    # DETAILS
    description = result.get("description", "")
    treatment = result.get("treatment", [])
    prevention = result.get("prevention", [])

    col1, col2 = st.columns(2)
    with col1:

        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.subheader("🦠 Disease Information")
        st.write(description if description else "No additional description available.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)

        st.subheader("💡 Recommended Solution")
        if treatment:
            st.write("**Treatment:**")
            for step in treatment:
                st.write(f"- {step}")
        if prevention:
            st.write("**Prevention:**")
            for tip in prevention:
                st.write(f"- {tip}")
        if not treatment and not prevention:
            st.write("No treatment needed — keep up the good care!")
        st.markdown("</div>", unsafe_allow_html=True)

# TAB 1 — UPLOAD IMAGE
with tab1:

    st.header("📤 Upload a Plant Image")
    st.write( "Choose a clear image of the plant leaf." )

    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.divider()

        col1, col2 = st.columns(2)
        # ORIGINAL IMAGE
        with col1:
            st.subheader("🌿 Your Plant")
            st.image(  image, width='stretch' )
        # ANALYZE BUTTON

        with col2:
            st.subheader("🤖 AI Analysis")
            st.write(
                "Click the button below to analyze the plant." )

            if st.button(
                "🔍 Detect Disease",  width='stretch' ):

                with st.spinner("Analyzing your plant..."):
                    result = analyze_image(
                        uploaded_file.getvalue(),
                        uploaded_file.name,
                        uploaded_file.type,
                    )
                display_result(result)

# TAB 2 — CAMERA
with tab2:

    st.header("📷 Take a Photo")
    st.write("Use your camera to take a picture of the plant leaf.")
    camera_photo = st.camera_input( "Take a picture of the plant" )

    if camera_photo is not None:

        image = Image.open(camera_photo)
        st.divider()
        col1, col2 = st.columns(2)
        # CAMERA IMAGE
        with col1:

            st.subheader("📸 Captured Image")
            st.image( image, width='stretch' )
        # ANALYSIS
        with col2:

            st.subheader("🤖 AI Analysis")
            if st.button(
                "🔍 Detect Disease",
                key="camera_detect",
                width='stretch'
            ):
                with st.spinner("Analyzing your plant..."):
                    result = analyze_image(
                        camera_photo.getvalue(),
                        camera_photo.name,
                        camera_photo.type,
                    )
                display_result(result)
# FOOTER
st.markdown(
    """
    <div class="footer">
        🌿 Plant Disease Detection System |
        TEAM 4 
    </div>
    """,
    unsafe_allow_html=True
)
