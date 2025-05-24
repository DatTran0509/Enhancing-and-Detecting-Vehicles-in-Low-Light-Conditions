from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from starlette.templating import Jinja2Templates
from PIL import Image
import io, base64, numpy as np

# Load models
from Zero_Dce import infer, enhance_net_nopool, load_checkpoint
from ultralytics import YOLO

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load Zero-DCE and YOLOv8 models
scale_factor = 1
DCE_net = enhance_net_nopool(scale_factor)
DCE_net = load_checkpoint(DCE_net, './models/Zero_DCE/Epoch99.pth').cpu()
yolo_model = YOLO('./models/yolov8m_enhanced_dce_And_denoise/yolov8m.pt').to('cpu')

# Convert PIL image to base64 for HTML rendering
def to_base64(image: Image.Image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "img_result": None})

@app.post("/process", response_class=HTMLResponse)
async def process_image(
    request: Request,
    file: UploadFile = File(...),
    mode: str = Form(...)
):
    contents = await file.read()
    original_image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Step 1: Enhance the image
    enhanced_np = infer(original_image, DCE_net, None, target_size=(640, 640))
    result_image = Image.fromarray(enhanced_np)

    # Step 2: Detect if user chose 'detect'
    if mode == "detect":
        results = yolo_model(np.ascontiguousarray(enhanced_np))
        detection_np = results[0].plot()
        result_image = Image.fromarray(detection_np)

    # Convert result to base64
    img_original_b64 = to_base64(original_image)
    img_result_b64 = to_base64(result_image)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "img_original": img_original_b64,
        "img_result": img_result_b64
    })
