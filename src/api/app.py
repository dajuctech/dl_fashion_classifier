from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API")

app = FastAPI(title="Fashion MNIST Classifier API")

# Load model on startup
MODEL_PATH = sorted([f for f in os.listdir("models") if f.endswith(".h5")], key=lambda x: os.path.getmtime(os.path.join("models", x)))[-1]
model = load_model(os.path.join("models", MODEL_PATH))
logger.info(f"Loaded model: {MODEL_PATH}")

@app.post("/predict")
async def predict(file: UploadFile):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("L")
        image = image.resize((28, 28))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=(0, -1))

        prediction = model.predict(image)
        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        logger.info(f"Prediction: {predicted_class}, Confidence: {confidence:.4f}")
        return {"class": predicted_class, "confidence": confidence}

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed. Please check your input file.")

@app.get("/")
async def root():
    return {"message": "Welcome to the Fashion MNIST Classifier API!"}

if __name__ == "__main__":
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)
