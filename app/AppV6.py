import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import torch
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import os
import base64
# ====================== تنظیمات صفحه ======================
st.set_page_config(
    page_title="تشخیص سرطان پستان با هوش مصنوعی" if st.session_state.get("lang", "FA") == "FA" else "Breast Cancer Detection with AI",
    page_icon="🎗️",
    layout="centered"
)

# ====================== انتخاب زبان ======================
lang = st.sidebar.selectbox("Language / زبان", ["FA", "EN"], key="language_selector")

# ====================== ترجمه‌ها (کامل و دقیق) ======================
translations = {
    "FA": {
        "home": "خانه",
        "about": "درباره ما",
        "title": "تشخیص سرطان پستان با هوش مصنوعی",
        "subtitle": "آپلود تصویر ماموگرافی و دریافت تحلیل خودکار",
        "upload_title": "آپلود تصویر ماموگرافی",
        "formats": "فرمت‌های مجاز: PNG, JPG, BMP, TIFF | حداکثر حجم: ۱۰ مگابایت",
        "no_file": "⚠️ هنوز فایلی انتخاب نشده است.",
        "start_analysis": "شروع تحلیل",
        "processing": "در حال پردازش تصویر و اجرای مدل هوش مصنوعی...",
        "uncertain": " هشدار: نتیجه تشخیص مدل نزدیک به مرز بوده و با عدم اطمینان همراه است. پیشنهاد می‌شود توسط پزشک متخصص بررسی شود.",
        "malignant": "سرطانی",
        "benign": "خوش‌خیم",
        "result": "نتیجه پیش‌بینی",
        "sample_caption": "تصویر نمونه از ماموگرافی دیجیتال",
        "footer": "دموی آموزشی تشخیص سرطان پستان • فقط برای اهداف تحقیقاتی و آموزشی",

        "about_title": "درباره ما",
        "about_text": "ما راه‌حل‌های انقلابی برای مشکلات صنعت ارائه می‌دهیم.",
        "mission_title": "ماموریت ما",
        "mission_text": "در Lando، متعهد هستیم بهترین خدمات را ارائه دهیم.",
        "team_title": "تیم ما",
        "team_sub": "با افرادی که پشت این محصول جادویی هستند آشنا شوید",

        # متن‌های سایدبار و مدل
        "model_settings": "تنظیمات مدل هوش مصنوعی",
        "current_path": "مسیر فعلی پروژه:",
        "default_found": "مدل پیش‌فرض پیدا شد:",
        "default_not_found": "مدل پیش‌فرض پیدا نشد.",
        "choose_method": "روش بارگذاری مدل",
        "use_default": "استفاده از مدل پیش‌فرض (best_model.pth)",
        "select_from_folder": "انتخاب مدل از پوشه models",
        "upload_new": "آپلود مدل جدید",
        "select_model": "مدل مورد نظر را انتخاب کنید",
        "no_model_in_folder": "هیچ فایل .pth یا .pt در پوشه models پیدا نشد!",
        "waiting_upload": "در انتظار آپلود مدل...",
        "model_uploaded": "مدل آپلود و ذخیره شد:",
        "model_loaded": "مدل با موفقیت لود شد:",
        "loading_model": "در حال بارگذاری مدل...",
        "error_loading": "خطا در بارگذاری مدل:",
    },
    "EN": {
        "home": "Home",
        "about": "About Us",
        "title": "Breast Cancer Detection with AI",
        "subtitle": "Upload a mammography image and receive automatic analysis",
        "upload_title": "Upload Mammography Image",
        "formats": "Allowed formats: PNG, JPG, BMP, TIFF | Max size: 10MB",
        "no_file": "⚠️ No file selected yet.",
        "start_analysis": "Start Analysis",
        "processing": "Processing image with AI model is running...",
        "uncertain": " Warning: The result is close to the decision boundary and comes with uncertainty. It is recommended to be reviewed by a specialist physician.",
        "malignant": "Malignant",
        "benign": "Benign",
        "result": "Prediction Result",
        "sample_caption": "Sample digital mammography image",
        "footer": "Breast cancer detection demo • For research and education only",

        "about_title": "About Us",
        "about_text": "We build revolutionary solutions for industry problems.",
        "mission_title": "Our Mission",
        "mission_text": "At Lando, we are committed to providing the best services.",
        "team_title": "Our Team",
        "team_sub": "Meet the people behind this magical product",

        "model_settings": "AI Model Settings",
        "current_path": "Current project path:",
        "default_found": "Default model found:",
        "default_not_found": "Default model not found.",
        "choose_method": "Model loading method",
        "use_default": "Use default model (best_model.pth)",
        "select_from_folder": "Select model from 'models' folder",
        "upload_new": "Upload new model",
        "select_model": "Select desired model",
        "no_model_in_folder": "No .pth or .pt files found in the 'models' folder!",
        "waiting_upload": "Waiting for model upload...",
        "model_uploaded": "Model uploaded and saved:",
        "model_loaded": "Model loaded successfully:",
        "loading_model": "Loading model...",
        "error_loading": "Error loading model:",
    }
}

t = translations[lang]  # برای دسترسی کوتاه

# ====================== CSS کاملاً داینامیک بر اساس زبان (طوسی / خاکستری) ======================
direction = "rtl" if lang == "FA" else "ltr"
text_align = "right" if lang == "FA" else "left"
opposite_align = "left" if lang == "FA" else "right"
font_family = "'Vazirmatn', sans-serif" if lang == "FA" else "'Vazirmatn', sans-serif, Arial, sans-serif"

# Palette: neutral / gray tones
# background: #f3f4f6 (gray-100)
# card bg: #ffffff
# primary text: #111827 (gray-900)
# secondary text: #374151 (slate-700)
# muted: #6b7280 (gray-500)
# accents & controls: #374151 -> darker on hover #0f172a

st.markdown(f"""
<style>
/* مخفی کردن هدر کامل Streamlit (شامل سه‌نقطه و Deploy) */
     header {{visibility: hidden;}}
    
    /* یا فقط دکمه‌های بالا راست */
     section+div>iframe {{display: none;}}
    
    /* مخفی کردن footer که نوشته "Streamlit" */
     footer {{visibility: hidden;}}
    
    /* اگر هنوز چیزی موند (جدیدترین نسخه Streamlit) */
     [data-testid="stHeader"] {{display: none !important;}}
     [data-testid="stDeployButton"] {{display: none !important;}}

@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {{
    font-family: {font_family};
    direction: {direction};
    text-align: {text_align};
    background-color: #f3f4f6; /* subtle gray background */
    color: #111827;
    margin: 0; padding: 0;
}}


.stMarkdown, .stText, h1, h2, h3, h4, h5, h6, p, div, span, label {{
    text-align: {text_align} !important;
    direction: {direction} !important;
    color: #374151; /* slate-700 for body text */
}}

.message-card {{
    border-radius: 12px; 
    padding: 18px 24px; 
    margin: 16px 0; 
    max-width: 700px; 
    box-shadow: 0 6px 20px rgba(15,23,42,0.06);
    background-color: #ffffff;
    transition: box-shadow 0.25s ease, transform 0.25s ease;
}}
.message-card:hover {{
    box-shadow: 0 12px 40px rgba(15,23,42,0.08);
    transform: translateY(-2px);
}}

.user-msg {{
    background: linear-gradient(135deg, #e6e9ee, #dbe0e6);
    color: #0f172a;
    text-align: {opposite_align};
    font-weight: 600;
    border: 1px solid rgba(55,65,81,0.06);
}}

.bot-msg {{
    background-color: #f8fafc;
    color: #0f172a;
    text-align: {text_align};
    font-weight: 500;
    border: 1px solid rgba(55,65,81,0.04);
}}

.top-nav {{
    background: transparent;
    padding: 12px 0;
    border-bottom: 1px solid #e6e7ea;
    margin-bottom: 40px;
    text-align: center;
    box-shadow: none;
    position: sticky;
    top: 0;
    z-index: 100;
    backdrop-filter: blur(6px);
}}

.top-nav .stRadio > div {{
    justify-content: center !important;
    gap: 40px;
}}

.top-nav label {{
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    color: #374151 !important;
    padding: 10px 22px !important;
    border-radius: 20px !important;
    cursor: pointer;
    transition: all 0.22s ease;
}}

.top-nav label:hover {{
    background-color: #eef2f6 !important;
    color: #111827 !important;
}}

.top-nav div[role="radiogroup"] > label[data-checked="true"] {{
    background-color: #374151 !important;
    color: white !important;
    box-shadow: 0 8px 24px rgba(55,65,81,0.08);
}}

div[data-testid="stButton"] > button {{
    background-color: #374151 !important; 
    color: white !important; 
    border-radius: 24px !important; 
    height: 3.4rem; 
    font-size: 1.02rem !important; 
    font-weight: 700; 
    width: auto !important; 
    padding: 0 22px; 
    border: none !important;
    box-shadow: 0 8px 22px rgba(15,23,42,0.08);
    transition: background-color 0.18s ease, transform 0.12s ease;
}}

div[data-testid="stButton"] > button:hover {{
    background-color: #0f172a !important;
    transform: translateY(-1px);
}}

.stFileUpload {{
    border: 2px dashed #e2e8f0; 
    border-radius: 18px; 
    padding: 16px 20px; 
    max-width: 700px; 
    margin: 0 auto 30px auto; 
    background-color: #ffffff;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}}

.stFileUpload:hover {{
    border-color: #cbd5e1;
    box-shadow: 0 8px 30px rgba(15,23,42,0.04);
}}

.team-circle-img {{
    width: 120px !important; 
    height: 120px; 
    object-fit: cover; 
    border-radius: 50%; 
    border: 3px solid #f3f4f6; 
    box-shadow: 0 10px 30px rgba(15,23,42,0.06); 
    transition: transform 0.35s ease, box-shadow 0.35s ease;
    margin-bottom: 12px;
}}

.team-circle-img:hover {{
    transform: scale(1.08);
    box-shadow: 0 18px 38px rgba(55,65,81,0.08);
}}

.team-member-name {{
    margin: 12px 0 6px 0 !important; 
    font-weight: 700; 
    font-size: 1.15rem; 
    color: #111827;
}}

.team-member-role {{
    color: #6b7280; 
    font-size: 0.98rem; 
    margin: 0;
}}

a.css-1vbkx1v, a.css-1c7y4gq, a.css-1wivap2,
div[data-testid="stHeading"] a, div[data-testid="stMarkdown"] h1 a,
div[data-testid="stMarkdown"] h2 a, div[data-testid="stMarkdown"] h3 a {{
    display: none !important;
}}

/* Responsive adjustments */
@media (max-width: 768px) {{
    .top-nav .stRadio > div {{
        gap: 20px;
        flex-wrap: wrap;
    }}
    .team-circle-img {{
        width: 90px !important;
        height: 90px;
    }}
    .message-card {{
        max-width: 100%;
        padding: 14px 16px;
    }}
    div[data-testid="stButton"] > button {{
        width: 100% !important;
        font-size: 1rem !important;
        height: 3rem;
        padding: 0;
    }}
    .stFileUpload {{
        max-width: 100%;
        padding: 12px 14px;
    }}
}}

/* Scrollbar styling for better UX */
::-webkit-scrollbar {{
    width: 8px;
}}

::-webkit-scrollbar-track {{
    background: #f3f4f6;
}}

::-webkit-scrollbar-thumb {{
    background: #6b7280;
    border-radius: 10px;
}}

::-webkit-scrollbar-thumb:hover {{
    background: #4b5563;
}}
.fixed-top-nav {{
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    background: rgba(255,255,255,0.75);
    border-bottom: 1px solid #e6e7ea;
    box-shadow: 0 6px 18px rgba(15,23,42,0.04);
    z-index: 9999;
    padding: 0;
    height: 68px;
    display: flex;
    align-items: center;
    backdrop-filter: blur(8px);
}}

.nav-container {{
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    width: 100%;
}}

.nav-logo {{
    font-weight: 700;
    font-size: 1.35rem;
    color: #374151;
}}

.nav-links {{
    display: flex;
    gap: 24px;
}}

.nav-item {{
    font-weight: 600;
    font-size: 1.05rem;
    color: #4b5563;
    cursor: pointer;
    padding: 8px 14px;
    border-radius: 10px;
    transition: all 0.22s ease;
}}

.nav-item:hover {{
    background-color: #f1f5f9;
    color: #111827;
}}

.nav-item.active {{
    background: linear-gradient(180deg, #374151, #2d3748);
    color: white !important;
    box-shadow: 0 8px 24px rgba(15,23,42,0.08);
}}

/* فاصله دادن به محتوای اصلی از نوار بالا */
.stApp > div:first-child {{
    margin-top: 90px !important;
}}


</style>
""", unsafe_allow_html=True)

# ====================== منوی بالای صفحه ======================
query_params = st.query_params
page = query_params.get("page", "home")
if lang == 'FA':
    st.markdown(f"""
    <div class="fixed-top-nav">
        <div class="nav-container">
            <div class="nav-logo">تشخیص سرطان پستان</div>
            <div class="nav-links">
                <a href="?page=home" class="nav-item {'active' if page == 'home' else ''}">{t['home']}</a>
                <a href="?page=about" class="nav-item {'active' if page == 'about' else ''}">{t['about']}</a>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="fixed-top-nav">
        <div class="nav-container">
            <div class="nav-logo">Breast Cancer Detection</div>
            <div class="nav-links">
                <a href="?page=home" class="nav-item {'active' if page == 'home' else ''}">{t['home']}</a>
                <a href="?page=about" class="nav-item {'active' if page == 'about' else ''}">{t['about']}</a>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# تنظیم صفحه واقعی بر اساس پارامتر URL
if page == "about":
    page = t["about"]
else:
    page = t["home"]  # پیش‌فرض


# ====================== توابع نمایش پیام ======================
def show_user_message(text):
    st.markdown(f"<div class='message-card user-msg'>{text}</div>", unsafe_allow_html=True)

def show_bot_message(text):
    st.markdown(f"<div class='message-card bot-msg'>{text}</div>", unsafe_allow_html=True)

# ====================== مسیر داینامیک مدل ======================
MODEL_DIR = os.path.join(os.getcwd(), "models")
DEFAULT_MODEL_NAME = "best_model.pth"
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, DEFAULT_MODEL_NAME)
os.makedirs(MODEL_DIR, exist_ok=True)

# سایدبار تنظیمات مدل
st.sidebar.markdown(f"### {t['model_settings']}")
st.sidebar.caption(f"{t['current_path']}\n`{os.getcwd()}`")

default_exists = os.path.exists(DEFAULT_MODEL_PATH)

if default_exists:
    st.sidebar.success(f"{t['default_found']}\n`{DEFAULT_MODEL_NAME}`")
else:
    st.sidebar.warning(t["default_not_found"])

# NOTE: small safety change: if default not found, show the use_default text with (not found)
use_default_label = t["use_default"] if default_exists else (t["use_default"] + " (not found)")

model_choice = st.sidebar.radio(
    t["choose_method"],
    [
        use_default_label,
        t["select_from_folder"],
        t["upload_new"]
    ],
    index=0 if default_exists else 1
)

model_path_to_load = None

if t["use_default"] in model_choice and default_exists:
    model_path_to_load = DEFAULT_MODEL_PATH

elif t["select_from_folder"] in model_choice:
    model_files = [f for f in os.listdir(MODEL_DIR) if f.lower().endswith(('.pth', '.pt'))]
    if not model_files:
        st.sidebar.error(t["no_model_in_folder"])
        st.stop()
    selected_model = st.sidebar.selectbox(t["select_model"], model_files)
    model_path_to_load = os.path.join(MODEL_DIR, selected_model)

else:  # آپلود مدل
    uploaded_model = st.sidebar.file_uploader(t["upload_new"], type=["pth", "pt"])
    if uploaded_model is not None:
        save_path = os.path.join(MODEL_DIR, uploaded_model.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_model.getbuffer())
        st.sidebar.success(f"{t['model_uploaded']}\n`{uploaded_model.name}`")
        model_path_to_load = save_path
    else:
        st.sidebar.info(t["waiting_upload"])
        st.stop()

# ====================== لود مدل ======================
@st.cache_resource(show_spinner=t["loading_model"])
def load_model(path: str):
    try:
        model = models.resnet18(pretrained=False)
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = torch.nn.Linear(512, 1)
        state_dict = torch.load(path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"{t['error_loading']} {e}")
        st.stop()

model = load_model(model_path_to_load)
st.sidebar.success(f"{t['model_loaded']}\n`{os.path.basename(model_path_to_load)}`")

# ====================== Grad-CAM ======================
@st.cache_resource
def get_gradcam_model(_model):
    return GradCAM(model=_model, target_layers=[_model.layer4[-1]])

cam = get_gradcam_model(model)

# ====================== پیش‌پردازش تصویر ======================
preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

def generate_gradcam(pil_image):
    gray_img = pil_image.convert("L")
    original_size = pil_image.size
    input_tensor = preprocess(gray_img).unsqueeze(0)
    grayscale_cam = cam(input_tensor=input_tensor)[0, :]
    gray_img_224 = np.array(gray_img.resize((224, 224))) / 255.0
    rgb_img_224 = np.stack([gray_img_224] * 3, axis=-1)
    visualization = show_cam_on_image(rgb_img_224, grayscale_cam, use_rgb=True)
    vis_image = Image.fromarray(visualization)
    return vis_image.resize(original_size, Image.LANCZOS)

def predict(image: Image.Image):
    tensor = preprocess(image)
    with torch.no_grad():
        output = model(tensor.unsqueeze(0))
        prob = torch.sigmoid(output).item()
    prediction = "Malignant" if prob > 0.5 else "Benign"
    return {"prediction": prediction, "confidence": prob}

def check_uncertainty(confidence):
    if 0.40 < confidence < 0.60:
        st.toast(t["uncertain"], icon="⚠️")

MAX_SIZE = 10 * 1024 * 1024  # 10MB
# تابع کمکی برای گرفتن اولین فایل تصویری از یک پوشه
PICTURE_DIR = os.path.join(os.getcwd(), "picture")
os.makedirs(PICTURE_DIR, exist_ok=True)
def get_first_image_from_folder(folder_name):
    folder_path = os.path.join(PICTURE_DIR, folder_name)
    if not os.path.exists(folder_path):
        return None
    for file in os.listdir(folder_path):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            return os.path.join(folder_path, file)
    return None
from PIL import Image, ImageOps


# ====================== صفحه خانه ======================
if page == t["home"]:
    st.markdown(f"<h1 style='text-align: center; color: #111827;'>{t['title']}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: #6b7280; font-size: 1.1rem;'>{t['subtitle']}</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### " + t["upload_title"])
    uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"], label_visibility="collapsed")
    st.markdown(f"<div style='text-align: center; color: #6b7280; font-size: 0.95rem; margin-top: -10px; margin-bottom: 20px;'>{t['formats']}</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        if uploaded_file.size > MAX_SIZE and lang!='FA':
            show_bot_message("⚠️File size exceeds 10MB")
            st.stop()
        if uploaded_file.size > MAX_SIZE and lang=='FA':
            show_bot_message("⚠️حجم فایل بیشتر از 10 مگابایت است")
            st.stop()

        image = Image.open(uploaded_file)
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")

        show_user_message(f"{uploaded_file.name}")

        if st.button(t["start_analysis"]):
            show_bot_message(t["processing"])
            result = predict(image)

            with st.spinner("Generating Grad-CAM heatmap..."):
                heatmap_image = generate_gradcam(image)

            check_uncertainty(result["confidence"])

            is_malignant = result["prediction"] == "Malignant"
            result_text = t["malignant"] if is_malignant else t["benign"]
            emoji = "🔴" if is_malignant else "🟢"

            show_bot_message(
                f"{emoji} **{t['result']}:** {result_text}<br>"
                f"Confidence: {result['confidence']:.1%}"
            )
            st.image(heatmap_image, caption="Grad-CAM Heatmap", use_container_width=True, clamp=True)

            if not is_malignant:
                st.balloons()

            with st.expander("Raw Output (JSON)"):
                st.json(result)

    else:
        show_bot_message(t["no_file"])
        st.image(
            get_first_image_from_folder("sample"),
            use_container_width=True,
            caption=t["sample_caption"]
        )

    st.markdown("---")
    st.markdown(f"<p style='text-align: center; color: #9ca3af; font-size: 0.9rem; margin-bottom: 2rem;'>{t['footer']}</p>", unsafe_allow_html=True)

# ====================== صفحه درباره ما ======================
else:
    st.markdown(f"""
    <div style="text-align: center; padding: 40px 20px; background: linear-gradient(180deg, #ffffff 0%, #f7f8fa 100%); border-radius: 20px; margin: 20px 0;">
        <h1 style="font-size: 3.2rem; font-weight: 700; color: #111827; margin-bottom: 16px;">{t['about_title']}</h1>
        <p style="font-size: 1.25rem; color: #374151; max-width: 800px; margin: 0 auto; line-height: 1.8;">
            {t['about_text']}
        </p>
    </div>
    """, unsafe_allow_html=True)
    logo_path = get_first_image_from_folder("logo")
    if logo_path is not None:
        with open(logo_path, "rb") as f:
            data = f.read()
        logo_base64 = base64.b64encode(data).decode()
    else:
        logo_base64 = None
    st.markdown(f"""
    <div style="display: flex; justify-content: center; gap: 20px; margin: 60px 0;">
        <img src="data:image/png;base64,{logo_base64}" width="600"/>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="text-align: center; margin: 80px 0 40px;">
        <h2 style="font-size: 2.8rem; font-weight: 700; color: #111827; margin-bottom: 30px;">{t['mission_title']}</h2>
        <p style="font-size: 1.22rem; color: #374151; max-width: 900px; margin: 0 auto; line-height: 1.9;">
            {t['mission_text']}
        </p>
        <hr style="width: 700px; margin: 40px auto; border: 1px solid #e6e7ea;">
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="text-align: center; margin: 80px 0 40px;">
        <h2 style="font-size: 2.8rem; font-weight: 700; color: #111827;">{t['team_title']}</h2>
        <p style="font-size: 1.2rem; color: #6b7280; margin-top: 10px;">
            {t['team_sub']}
        </p>
    </div>
    """, unsafe_allow_html=True)
    tatari_path = get_first_image_from_folder("tatari")
    if tatari_path is not None:
        with open(tatari_path, "rb") as f:
            data = f.read()
        tatari_base64 = base64.b64encode(data).decode()
        
    golibidgoli_path = get_first_image_from_folder("golibidgoli")
    if golibidgoli_path is not None:
        with open(golibidgoli_path, "rb") as f:
            data = f.read()
        golibidgoli_base64 = base64.b64encode(data).decode()
    moharrampour_path = get_first_image_from_folder("moharrampour")
    if moharrampour_path is not None:
        with open(moharrampour_path, "rb") as f:
            data = f.read()
        moharrampour_base64 = base64.b64encode(data).decode()
    shams_path = get_first_image_from_folder("shams")
    if shams_path is not None:
        with open(shams_path, "rb") as f:
            data = f.read()
        shams_base64 = base64.b64encode(data).decode()
    team_members = [
        {"name": "محمد مهدی ترک‌تتاری" if lang == "FA" else "MohammadMahdi TorkTatari", "role": "توسعه‌دهنده وب" if lang == "FA" else "Web Developer", "image": f"data:image/png;base64,{tatari_base64}"},
        {"name": "محمد هادی گلی بیدگلی" if lang == "FA" else "MohammadHadi GoliBidGoli", "role": "دانشمند داده" if lang == "FA" else "Data Scientist", "image": f"data:image/png;base64,{golibidgoli_base64}"},
        {"name": "علیرضا شمس" if lang == "FA" else "Alireza Shams", "role": "دانشمند داده" if lang == "FA" else "Data Scientist", "image": f"data:image/png;base64,{shams_base64}"},
        {"name": "الهه محرم‌پور" if lang == "FA" else "Elahe MoharramPour", "role": "دانشمند داده" if lang == "FA" else "Data Scientist", "image": f"data:image/png;base64,{moharrampour_base64}"},
    ]

    for i in range(0, len(team_members), 4):
        cols = st.columns(4)
        for idx, col in enumerate(cols):
            if i + idx < len(team_members):
                member = team_members[i + idx]
                with col:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 20px;">
                        <img src="{member['image']}" class="team-circle-img" alt="{member['name']}">
                        <p class="team-member-name">{member['name']}</p>
                        <p class="team-member-role">{member['role']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        st.markdown("<br><br>", unsafe_allow_html=True) 