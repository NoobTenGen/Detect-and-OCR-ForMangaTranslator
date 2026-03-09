import io
import os
os.environ['FLAGS_use_onednn'] = '0'
os.environ['FLAGS_use_onednn_v2'] = '0'
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
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
                font = ImageFont.truetype(font_name, 28)
                break
            except:
                continue
        
        if font is None:
            font = ImageFont.load_default()
        
        colors = [
            (139, 0, 0),      # 深红色
            (0, 0, 139),      # 深蓝色
            (0, 100, 0),      # 深绿色
            (184, 134, 11),   # 深橙色
            (75, 0, 130),     # 深紫色
            (0, 139, 139),    # 深青色
            (139, 0, 139),    # 深洋红色
            (184, 134, 11)    # 深黄色
        ]
        
        def wrap_text(text, font, max_width):
            lines = []
            current_line = ""
            
            for char in text:
                test_line = current_line + char
                bbox = draw.textbbox((0, 0), test_line, font=font)
                text_width = bbox[2] - bbox[0]
                
                if text_width <= max_width:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = char
            
            if current_line:
                lines.append(current_line)
            
            return lines
        
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
                box_width = right - left
                
                lines = wrap_text(text, font, box_width)
                line_height = 32
                
                for line_idx, line in enumerate(lines):
                    draw.text((left, top - (len(lines) - line_idx) * line_height - 5), line, fill=color, font=font)
        
        img_byte_arr = io.BytesIO()
        img_draw.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return StreamingResponse(img_byte_arr, media_type="image/png")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
