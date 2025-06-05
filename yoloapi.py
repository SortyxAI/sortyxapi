from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import numpy as np
from io import BytesIO
from PIL import Image

app = FastAPI()
model = YOLO("yolov8n.pt")

@app.post("/classify")
async def predict(file: UploadFile = File(...)):
    # Read image file
    image_data = await file.read()
    image = Image.open(BytesIO(image_data))
    image = np.array(image)

    # Run inference
    results = model(image)

    label = results[0].names[int(results[0].boxes.cls[0])] if results[0].boxes else "Unknown"
    confidence = results[0].boxes.conf[0] if results[0].boxes else 0.0
    confidence = int(confidence * 100)
    confidence = min(max(confidence, 0), 100)

    # Return label and confidence
    return {"label": label, "confidence": confidence}