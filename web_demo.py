import io
import os
import base64
os.environ['FLAGS_use_onednn'] = '0'
os.environ['FLAGS_use_onednn_v2'] = '0'
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from paddleocr import TextDetection
from PIL import Image, ImageDraw, ImageFont
from config import DETECTOR_MODEL_NAME, DETECTOR_MODEL_PATH, DETECTOR_MODEL_DEVICE, OCR_MODEL_NAME, OCR_MODEL_PATH, OCR_MODEL_DEVICE
from manga_ocr import MangaOcr
import json
import cv2

app = FastAPI()

templates = Jinja2Templates(directory="templates")

detector = TextDetection(
    model_name=DETECTOR_MODEL_NAME, 
    model_dir=DETECTOR_MODEL_PATH, 
    device=DETECTOR_MODEL_DEVICE,
    enable_mkldnn=False
)
mocr = MangaOcr(pretrained_model_name_or_path=OCR_MODEL_PATH)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process")
async def process_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_array = np.array(img)
        
        if not img_array.any():
            raise HTTPException(status_code=400, detail="Failed to process image")
        
        output = detector.predict(img_array)
        
        if not output:
            raise HTTPException(status_code=400, detail="Failed to detect text")
        
        detections = []
        for det in output:
            detection = {
                'dt_polys': det['dt_polys'].tolist() if 'dt_polys' in det else [],
                'dt_scores': det['dt_scores'] if 'dt_scores' in det else []
            }
            detections.append(detection)
        
        poly_texts = []
        for detection in detections:
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
                    text = mocr(cropped)
                    poly_texts.append({'poly': poly, 'text': text})
        
        img_draw = img.copy()
        draw = ImageDraw.Draw(img_draw)
        
        font = None
        japanese_fonts = [
            "msgothic.ttc",
            "msmincho.ttc",
            "yugothic.ttc",
            "yu-mincho.ttc",
            "meiryo.ttc",
            "simsun.ttc"
        ]
        
        for font_name in japanese_fonts:
            try:
                font = ImageFont.truetype(font_name, 32)
                break
            except:
                continue
        
        if font is None:
            font = ImageFont.load_default()
        
        colors = [
            (139, 0, 0),
            (0, 0, 139),
            (0, 100, 0),
            (184, 134, 11),
            (75, 0, 130),
            (0, 139, 139),
            (139, 0, 139),
            (184, 134, 11)
        ]
        
        results = []
        for idx, poly_text in enumerate(poly_texts):
            poly = poly_text['poly']
            text = poly_text['text']
            if len(poly) >= 4:
                color = colors[idx % len(colors)]
                draw.polygon([(point[0], point[1]) for point in poly], outline=color, width=3)
                
                x_coords = [point[0] for point in poly]
                y_coords = [point[1] for point in poly]
                left = min(x_coords)
                right = max(x_coords)
                top = min(y_coords)
                bottom = max(y_coords)
                
                center_x = (left + right) / 2
                center_y = (top + bottom) / 2
                
                circle_radius = 20
                draw.ellipse([center_x - circle_radius, center_y - circle_radius, 
                             center_x + circle_radius, center_y + circle_radius], 
                            fill=color, outline=color)
                
                text_bbox = draw.textbbox((0, 0), str(idx + 1), font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                text_x = center_x - text_width / 2
                text_y = center_y - text_height / 2
                draw.text((text_x, text_y), str(idx + 1), fill=(255, 255, 255), font=font)
                
                results.append({
                    'index': idx + 1,
                    'poly': poly,
                    'text': text,
                    'color': color
                })
        
        img_byte_arr = io.BytesIO()
        img_draw.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        result_image_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        
        original_byte_arr = io.BytesIO()
        img.save(original_byte_arr, format='PNG')
        original_byte_arr.seek(0)
        original_image_base64 = base64.b64encode(original_byte_arr.getvalue()).decode('utf-8')
        
        return JSONResponse({
            'original_image': original_image_base64,
            'result_image': result_image_base64,
            'results': results
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
