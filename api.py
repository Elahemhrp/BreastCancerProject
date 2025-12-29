# api.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch
import numpy as np
import base64
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms

# ایمپورت کردن ماژول‌های پروژه خودتان
from core.inference import Predictor
from core.preprocessing import preprocess_image
from core.config import Config

app = FastAPI(title="Breast Cancer API")

# --- تنظیمات CORS (حیاتی برای اتصال به React) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # در محیط واقعی باید آدرس سایت را بگذارید، ولی برای تست * عالی است
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- راه‌اندازی مدل ---
print("Initializing Model...")
# مدل را یک بار لود می‌کنیم تا سرعت بالا برود
predictor = Predictor(model_path="checkpoints/best_model.pth") 

# اگر مدل لود نشد (Mock Mode)، وارنینگ می‌دهیم
if predictor.mock_mode:
    print("WARNING: Running in MOCK MODE (No model found).")

# --- تنظیمات Grad-CAM ---
# توجه: این بخش فرض می‌کند مدل شما لایه‌ای به نام layer4 دارد (مثل ResNet)
# اگر از EfficientNet استفاده می‌کنید باید layer مناسب را انتخاب کنید.
target_layer = None
if not predictor.mock_mode and predictor.model:
    try:
        # بررسی می‌کنیم آیا مدل داخل یک wrapper به نام backbone است؟
        if hasattr(predictor.model, 'backbone'):
            # دسترسی به لایه آخر ResNet که داخل backbone است
            # معمولاً layer4 آخرین لایه کانولوشنی ResNet است
            target_layer = predictor.model.backbone.layer4[-1]
        else:
            # حالت استاندارد (اگر مدل مستقیم ResNet باشد)
            target_layer = predictor.model.layer4[-1]
            
        cam = GradCAM(model=predictor.model, target_layers=[target_layer])
        print(f"Grad-CAM initialized successfully on layer: {target_layer}")
    except Exception as e:
        print(f"Error initializing Grad-CAM: {e}")
        cam = None
else:
    cam = None

# --- توابع کمکی ---
def image_to_base64(image: Image.Image) -> str:
    """تبدیل تصویر PIL به رشته Base64 برای ارسال به فرانت‌اند"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def generate_heatmap(pil_image, model, cam_obj):
    """تولید تصویر هیت‌مپ"""
    try:
        # 1. آماده‌سازی تنسور ورودی
        # نکته کلیدی: استفاده از تابع اصلی پروژه برای اطمینان از هماهنگی کانال‌ها (RGB) و نرمال‌سازی
        input_tensor = preprocess_image(pil_image).to(predictor.device)
        
        # 2. تولید ماسک Grad-CAM
        # targets=None یعنی کلاس پیش‌بینی شده را به صورت خودکار انتخاب کن
        grayscale_cam = cam_obj(input_tensor=input_tensor, targets=None)[0, :]
        
        # 3. آماده‌سازی تصویر پس‌زمینه برای نمایش
        # تصویر را ریسایز می‌کنیم تا با ابعاد تنسور (224x224) یکی شود
        img_resized = pil_image.resize((224, 224))
        img_np = np.array(img_resized)
        
        # اگر تصویر ورودی سیاه و سفید است، باید به RGB تبدیل شود تا بتوانیم هیت‌مپ رنگی روی آن بکشیم
        if len(img_np.shape) == 2:
            # تبدیل Grayscale به RGB با تکرار کانال‌ها
            img_np = np.stack([img_np] * 3, axis=-1)
        elif img_np.shape[2] == 1:
            img_np = np.concatenate([img_np, img_np, img_np], axis=-1)
            
        # نرمال‌سازی بین 0 و 1 (الزامی برای show_cam_on_image)
        img_float = img_np.astype(np.float32) / 255.0
        
        # ترکیب هیت‌مپ با تصویر اصلی
        visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
        return Image.fromarray(visualization)
        
    except Exception as e:
        print(f"Grad-CAM Error: {e}")
        import traceback
        traceback.print_exc() # چاپ خطای کامل برای دیباگ دقیق‌تر
        return None

# --- Endpoint اصلی ---
@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    # 1. خواندن فایل آپلود شده
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # 2. دریافت پیش‌بینی (کلاس و درصد اطمینان)
    result = predictor.predict(image)
    
    # 3. تولید هیت‌مپ (فقط اگر مدل واقعی باشد)
    heatmap_b64 = None
    if not result.get("mock", False) and cam:
        heatmap_img = generate_heatmap(image, predictor.model, cam)
        if heatmap_img:
            heatmap_b64 = image_to_base64(heatmap_img)
    
    # 4. آماده‌سازی خروجی نهایی JSON
    response = {
        "class": result['class'],
        "confidence": result['confidence'],
        "yellow_flag": result['yellow_flag'],
        "heatmap_base64": heatmap_b64,
        "is_mock": result.get("mock", False)
    }
    
    return response

# تست ساده برای اطمینان از روشن بودن سرور
@app.get("/")
def read_root():
    return {"status": "Online", "message": "Breast Cancer API is running"}