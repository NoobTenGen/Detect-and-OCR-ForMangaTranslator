# Paddle OCR Service

基于FastAPI的OCR文本检测与识别服务，集成了PaddleOCR和MangaOCR模型，提供文本检测和识别的REST API接口。

## 功能特性

- 文本检测：使用PaddleOCR的PP-OCRv5模型进行文本区域检测
- 文本识别：使用MangaOCR模型进行文本内容识别
- REST API：提供简洁的HTTP接口
- 异步处理：基于FastAPI的异步请求处理
- 图片下载：支持通过URL下载和处理图片

## 模型说明

### PaddleOCR模型
基于PP-OCRv5_mobile_det根据数据集Manga109-s进行微调，在测试集上对文本区域检测的准确率为85%。

### MangaOCR模型
本项目使用的MangaOCR模型来自开源项目 [kha-white/manga-ocr](https://github.com/kha-white/manga-ocr)。

**项目简介**：
MangaOCR是一个专门用于日语文本识别的OCR工具，主要针对日本漫画场景进行了优化。它使用基于Transformers的Vision Encoder Decoder框架构建的自定义端到端模型。

**主要特性**：
- 支持垂直和水平文本识别
- 能够处理带有振假名（furigana）的文本
- 支持文本叠加在图像上的场景
- 对各种字体和字体样式具有良好的适应性
- 能够处理低质量图像
- 支持在单次前向传播中识别多行文本，无需将文本气泡分割成多行

**适用场景**：
- 日本漫画文本识别
- 日语印刷文本识别（如小说、游戏等）
- 各种印刷日语文本的通用OCR

**模型信息**：
- 模型名称：manga-ocr-base
- 开源地址：https://github.com/kha-white/manga-ocr
- 模型大小：约400MB
- 首次运行时会自动下载模型文件

## 技术栈

- **Web框架**: FastAPI
- **OCR引擎**: PaddleOCR, MangaOCR
- **图像处理**: Pillow, OpenCV
- **深度学习**: PyTorch, PaddlePaddle
- **HTTP客户端**: httpx

## 项目结构

```
paddle_ocr/
├── main.py           # 主程序入口
├── config.py         # 配置文件
├── requirements.txt  # 依赖包列表
├── .gitignore       # Git忽略文件
└── README.md        # 项目说明
```

## 安装说明

### 环境要求

- Python 3.10+
- Conda环境

### 安装步骤

1. 克隆项目到本地

2. 创建并激活conda环境（如果还没有环境）：
```bash
conda create -n paddle_ocr python=3.10
conda activate paddle_ocr
```

3. 安装依赖包：
```bash
pip install -r requirements.txt
```

4. 下载模型文件：
- 文本检测模型：PP-OCRv5_mobile_det
- OCR识别模型：manga-ocr-base

将模型文件放置在对应的目录中：
```
models/
├── detector/
│   └── PP-OCRv5_mobile_det/
└── ocr/
    └── manga_ocr/
```

## 使用方法

### 启动服务

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### API接口

#### 1. 文本检测

**接口**: `POST /detect`

**请求参数**:
```json
{
  "url": "图片URL地址"
}
```

**返回结果**:
```json
{
  "detections": [
    {
      "dt_polys": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
      "dt_scores": 0.95
    }
  ]
}
```

#### 2. 文本识别

**接口**: `POST /ocr`

**请求参数**:
```json
{
  "url": "图片URL地址",
  "detections": "文本检测结果JSON字符串"
}
```

**返回结果**:
```json
{
  "detections": {
    "dt_polys": [...],
    "dt_scores": [...],
    "dt_text": ["识别的文本1", "识别的文本2"]
  }
}
```

### 使用示例

#### Python示例

```python
import requests
import json

# 文本检测
detect_url = "http://localhost:8000/detect"
detect_response = requests.post(detect_url, json={
    "url": "https://example.com/image.jpg"
})
detections = detect_response.json()

# 文本识别
ocr_url = "http://localhost:8000/ocr"
ocr_response = requests.post(ocr_url, json={
    "url": "https://example.com/image.jpg",
    "detections": json.dumps(detections)
})
result = ocr_response.json()
print(result)
```

#### curl示例

```bash
# 文本检测
curl -X POST "http://localhost:8000/detect" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/image.jpg"}'

# 文本识别
curl -X POST "http://localhost:8000/ocr" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/image.jpg", "detections": "{\"detections\": [...]}"}'
```

## 配置说明

在 `config.py` 中可以配置模型路径和设备：

```python
# 文本检测模型配置
DETECTOR_MODEL_NAME = "PP-OCRv5_mobile_det"
DETECTOR_MODEL_PATH = "./models/detector/PP-OCRv5_mobile_det"
DETECTOR_MODEL_DEVICE = "cpu"  # 可选: "cpu" 或 "cuda"

# OCR识别模型配置
OCR_MODEL_NAME = "kha-white/manga-ocr-base"
OCR_MODEL_PATH = "./models/ocr/manga_ocr"
OCR_MODEL_DEVICE = "cpu"  # 可选: "cpu" 或 "cuda"
```

## 注意事项

1. 首次运行时需要下载模型文件，可能需要较长时间
2. 图片URL需要是可公开访问的地址
3. 建议使用GPU加速以提高处理速度
4. 确保网络连接正常，用于下载图片和模型

## 许可证

本项目仅供学习和研究使用。

## 贡献

欢迎提交Issue和Pull Request！
