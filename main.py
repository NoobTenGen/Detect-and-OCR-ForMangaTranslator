import io
import os
os.environ['FLAGS_use_onednn'] = '0'
os.environ['FLAGS_use_onednn_v2'] = '0'
import httpx
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from paddleocr import TextDetection
from PIL import Image
from config import DETECTOR_MODEL_NAME, DETECTOR_MODEL_PATH, DETECTOR_MODEL_DEVICE
from manga_ocr import MangaOcr
from contextlib import asynccontextmanager
from config import OCR_MODEL_NAME, OCR_MODEL_PATH, OCR_MODEL_DEVICE
from contextlib import asynccontextmanager
from manga_ocr import MangaOcr
from PIL import Image
import json


@asynccontextmanager
async def lifespan(app: FastAPI):
    global client
    client = httpx.AsyncClient()
    yield
    await client.close()


app = FastAPI(lifespan=lifespan)

detector = TextDetection(
    model_name=DETECTOR_MODEL_NAME, 
    model_dir=DETECTOR_MODEL_PATH, 
    device=DETECTOR_MODEL_DEVICE,
    enable_mkldnn=False
    )
mocr = MangaOcr(pretrained_model_name_or_path=OCR_MODEL_PATH)

class DetectorRequest(BaseModel):
    url: str

class OCRRequest(BaseModel):
    url: str
    detections: str

@app.post("/detect")
async def process_detect(task: DetectorRequest):
    # 1. 下载图片
    response = await client.get(task.url, timeout=10.0)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to fetch image")

    # 2. 图片预处理
    img = Image.open(io.BytesIO(response.content)).convert("RGB")
    img_array = np.array(img)
    if not img_array.any():
        raise HTTPException(status_code=400, detail="Failed to process image")
    # 3. 模型推理
    output = detector.predict(img_array)
    if not output:
        raise HTTPException(status_code=400, detail="Failed to detect text")
    
    if output is not None and len(output) > 0:
        detections = []
        for det in output:
            detection = {
                'dt_polys': det['dt_polys'].tolist() if 'dt_polys' in det else [],
                'dt_scores': det['dt_scores'] if 'dt_scores' in det else []
            }
            detections.append(detection)
    # 4. 返回结果
    return {"detections": detections}

@app.post("/ocr")
async def process_ocr(task: OCRRequest):
    # 1.1 下载图片
    img_url = await client.get(task.url, timeout=10.0)
    if img_url.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to fetch image")
    
    # 1.2 下载detections
    detections_response = task.detections

    detections = json.loads(detections_response)

    # 2. 图片预处理
    img = Image.open(io.BytesIO(img_url.content)).convert("RGB")
    # 3. 根据detection裁剪图片
    cropped_images = []
    for detection in detections.get("detections", []):
        polys = detection.get("dt_polys", [])
        for poly in polys:
            if len(poly) >= 4:
                x_coords = [point[0] for point in poly]
                y_coords = [point[1] for point in poly]
                left = min(x_coords)
                top = min(y_coords)
                right = max(x_coords)
                bottom = max(y_coords)
                cropped = img.crop((left, top, right, bottom))
                cropped_images.append(cropped)
    detections["dt_text"] = []
    for cropped in cropped_images:
        text = mocr(cropped)
        detections["dt_text"].append(text)
    # 4. 返回结果
    return {"detections": detections}
