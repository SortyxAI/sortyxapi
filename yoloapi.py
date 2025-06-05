from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

app = FastAPI()
model = YOLO("yolov8n.pt")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image file
    image_data = await file.read()
    image = Image.open(BytesIO(image_data))
    image = np.array(image)

    # Run inference
    results = model(image)

    # Process results (example: return bounding boxes as JSON)
    boxes = results[0].boxes.xyxy.cpu().numpy().tolist()  # Extract bounding boxes
    return {"boxes": boxes}