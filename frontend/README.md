# 🧠 Breast Cancer Detection - Backend API

این بخش شامل هسته پردازشی پروژه تشخیص سرطان پستان است که با زبان **Python** و فریم‌ورک **FastAPI** توسعه داده شده است. وظیفه این بخش، دریافت تصاویر، پیش‌پردازش، اجرای مدل هوش مصنوعی (ResNet/EfficientNet) و تولید خروجی‌های تفسیرپذیر (Grad-CAM) است.

## 🛠️ تکنولوژی‌های استفاده شده (Tech Stack)

* **Web Framework:** [FastAPI](https://fastapi.tiangolo.com/) (برای ساخت API سریع و مدرن)
* **Deep Learning:** [PyTorch](https://pytorch.org/) & [Torchvision](https://pytorch.org/vision/stable/index.html)
* **Image Processing:** [OpenCV](https://opencv.org/) & [Pillow](https://python-pillow.org/)
* **Explainable AI:** [Grad-CAM](https://github.com/jacobgil/pytorch-grad-cam) (برای تولید نقشه‌های حرارتی)
* **Server:** [Uvicorn](https://www.uvicorn.org/) (سرور ASGI)

## 🚀 راهنمای نصب و اجرا (Installation & Run)

پیش‌نیاز: مطمئن شوید که **Python 3.8+** روی سیستم شما نصب است.

### ۱. ساخت محیط مجازی (اختیاری ولی پیشنهادی)
```bash
python -m venv venv
# فعال‌سازی در ویندوز:
venv\Scripts\activate
# فعال‌سازی در مک/لینوکس:
source venv/bin/activate
├── checkpoints/             # 💾 فایل‌های مدل آموزش دیده (.pth)
├── data/                    # داده‌های ورودی و CSVها
└── requirements.txt         # لیست کتابخانه‌های پایتون
