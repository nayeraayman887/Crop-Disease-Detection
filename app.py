import streamlit as st
from PIL import Image
import requests

# ============================================================
# BACKEND CONNECTION
# ============================================================
# While testing locally: run the backend first (uvicorn main:app --port 8000),
# then this points at it here.
# Once the backend is deployed (e.g. Hugging Face Spaces), swap this for that
# URL, e.g. "https://your-username-your-space.hf.space"
BACKEND_URL = "http://localhost:8000"


def analyze_image(file_bytes: bytes, filename: str, content_type: str) -> dict:
    """Sends the image to the FastAPI backend's /predict endpoint and
    returns the parsed JSON result (or an 'error' status dict if the
    backend couldn't be reached)."""
    try:
        files = {"file": (filename, file_bytes, content_type)}
        response = requests.post(f"{BACKEND_URL}/predict", files=files, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {
            "status": "error",
            "message": f"Could not reach the backend at {BACKEND_URL}. "
                       f"Make sure it's running (uvicorn main:app --port 8000).",
        }
    except requests.exceptions.HTTPError as e:
        return {"status": "error", "message": f"Backend returned an error: {e}"}
    except Exception as e:
        return {"status": "error", "message": f"Unexpected error: {e}"}


# ============================================================
# PAGE CONFIGURATION

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)
# CUSTOM CSS
# ============================================================

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
# ============================================================

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

    # --------------------------------------------------------
    # Backend couldn't be reached, or something else went wrong
    # --------------------------------------------------------
    if status == "error":
        st.error(result.get("message", "Something went wrong analyzing the image."))
        return

    # --------------------------------------------------------
    # Model wasn't confident enough (below CONFIDENCE_THRESHOLD)
    # --------------------------------------------------------
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
        st.error(result.get("message", "Unexpected response from the backend."))
        return

    disease_name = result.get("disease")
    confidence = result.get("confidence", 0) * 100

    # --------------------------------------------------------
    # Backend predicted a class but disease_info.json has no entry for it
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # STATUS — normal healthy / diseased result
    # --------------------------------------------------------
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
            st.image(  image, use_container_width=True )
        # ANALYZE BUTTON

        with col2:

            st.subheader("🤖 AI Analysis")
            st.write(
                "Click the button below to analyze the plant." )

            if st.button(
                "🔍 Detect Disease",  use_container_width=True ):

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
        # ----------------------------------------------------

        with col1:

            st.subheader("📸 Captured Image")
            st.image( image, use_container_width=True )
        # ANALYSIS
        with col2:

            st.subheader("🤖 AI Analysis")
            if st.button(
                "🔍 Detect Disease",
                key="camera_detect",
                use_container_width=True
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