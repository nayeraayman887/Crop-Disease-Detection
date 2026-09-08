import json

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import config
import model_utils

app = FastAPI(title="Plant Disease Detection API")

# Allows your frontend (running on a different port/domain) to call this API.
# For a real deployment, replace "*" with your actual frontend URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("disease_info.json", "r", encoding="utf-8") as f:
    DISEASE_INFO = json.load(f)


@app.on_event("startup")
def startup_event():
    # Load the model once when the server starts, not on every request.
    model_utils.load_model()
    print("Model loaded and ready.")


@app.get("/")
def root():
    return {"status": "Plant Disease Detection API is running"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    image_bytes = await file.read()

    try:
        class_name, confidence = model_utils.predict(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    if confidence < config.CONFIDENCE_THRESHOLD:
        return {
            "status": "uncertain",
            "message": "Could not confidently identify a disease. Try a clearer, closer photo of the affected leaf.",
            "confidence": round(confidence, 4),
        }

    info = DISEASE_INFO.get(class_name)
    if info is None:
        # This means CLASS_NAMES in config.py doesn't match disease_info.json keys.
        return {
            "status": "success",
            "raw_class": class_name,
            "confidence": round(confidence, 4),
            "message": "Prediction succeeded but no treatment info is on file for this class. "
                       "Add an entry for it in disease_info.json.",
        }

    return {
        "status": "success",
        "disease": info["common_name"],
        "confidence": round(confidence, 4),
        "description": info["description"],
        "treatment": info["treatment"],
        "prevention": info["prevention"],
    }
