from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
import numpy as np
import cv2

from enhance import enhance_array
from detect import detect_vehicles

from prometheus_client import Counter, Histogram, generate_latest

app = FastAPI()

REQUEST_COUNT = Counter('request_count', 'Total Requests')
PROCESS_TIME = Histogram('process_time_seconds', 'Processing Time')

@app.get("/")
def home():
    return {"message": "Fog + YOLO API Running"}

@app.post("/process")
@PROCESS_TIME.time()
async def process_image(file: UploadFile = File(...)):
    REQUEST_COUNT.inc()

    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    enhanced = enhance_array(img)
    detected, count = detect_vehicles(enhanced)

    _, encoded = cv2.imencode('.jpg', detected)

    return Response(
        content=encoded.tobytes(),
        media_type="image/jpeg",
        headers={"X-Vehicle-Count": str(count)}
    )

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")