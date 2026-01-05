import streamlit as st
import sys
import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

# افزودن مسیر ریشه پروژه برای ایمپورت‌های core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ایمپورت ماژول‌های استاندارد پروژه
from core.inference import Predictor
from core.config import Config
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ====================== 1. CONFIG & STYLES ======================
st.set_page_config(
    page_title="MicroCalc AI Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تعریف رنگ‌ها و استایل‌های فایل HTML شما
st.markdown("""
<style>
    /* فونت و رنگ‌بندی کلی */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #0f172a; /* Navy */
    }
    
    .stApp {
        background-color: #f8f6f6; /* Background Light */
    }

    /* حذف هدر پیش‌فرض استریم‌لیت */
    header {visibility: hidden;}
    .block-container {padding-top: 2rem;}

    /* استایل سایدبار */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        box-shadow: 4px 0 24px rgba(0,0,0,0.05);
        border-right: 1px solid #e2e8f0;
    }

    /* دکمه اصلی (صورتی) */
    .stButton > button {
        background-color: #ee2b5b !important;
        color: white !important;
        border-radius: 0.75rem !important;
        padding: 0.75rem 1rem !important;
        font-weight: 700 !important;
        border: none !important;
        box-shadow: 0 10px 15px -3px rgba(238, 43, 91, 0.3);
        width: 100%;
        transition: all 0.2s ease-in-out;
    }
    .stButton > button:hover {
        background-color: #d61f4b !important;
        transform: scale(0.98);
    }

    /* کارت‌های وضعیت (Grid Cards) */
    .stat-card {
        background-color: white;
        padding: 1.25rem;
        border-radius: 0.75rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    .stat-label {
        color: #64748b;
        font-size: 0.875rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    .stat-value {
        color: #0f172a;
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: -0.025em;
    }

    /* کارت نتیجه (Prediction Card) */
    .result-card {
        background-color: white;
        border-radius: 1rem;
        border: 1px solid #e2e8f0;
        overflow: hidden;
        margin-bottom: 1.5rem;
    }
    .result-header {
        background-color: rgba(248, 250, 252, 0.8);
        padding: 1.25rem 1.5rem;
        border-bottom: 1px solid #f1f5f9;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .result-body {
        padding: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ====================== 2. LOAD CORE LOGIC ======================
@st.cache_resource
def load_backend():
    # استفاده از کلاس Predictor استاندارد شما
    predictor = Predictor() # مدل را لود می‌کند (یا به حالت Mock می‌رود)
    
    # آماده‌سازی Grad-CAM (نیاز به دسترسی مستقیم به مدل دارد)
    if not predictor.mock_mode:
        # فرض بر این است که مدل شما ResNet است. اگر EfficientNet است، لایه هدف فرق می‌کند.
        target_layer = predictor.model.backbone.layer4[-1] 
        cam = GradCAM(model=predictor.model, target_layers=[target_layer])
    else:
        cam = None
        
    return predictor, cam

predictor, cam = load_backend()

# تابع تولید Grad-CAM
def generate_gradcam(pil_image, cam_obj):
    if cam_obj is None: return None
    
    # پیش‌پردازش مشابه ترینینگ
    img_tensor = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])(pil_image.convert("RGB")).unsqueeze(0)
    
    # تولید نقشه
    grayscale_cam = cam_obj(input_tensor=img_tensor)[0, :]
    
    # ترکیب با تصویر اصلی
    img_resized = np.array(pil_image.resize((224, 224)).convert("RGB")) / 255.0
    visualization = show_cam_on_image(img_resized, grayscale_cam, use_rgb=True)
    return Image.fromarray(visualization)

# ====================== 3. SIDEBAR UI ======================
with st.sidebar:
    st.markdown("""
    <div style="padding: 1rem 0; border-bottom: 1px solid #f1f5f9; margin-bottom: 1.5rem;">
        <h2 style="font-size: 1.25rem; font-weight: 700; color: #0f172a; display: flex; align-items: center; gap: 0.5rem;">
            <span style="color: #ee2b5b;">📂</span> Patient Data
        </h2>
        <p style="font-size: 0.875rem; color: #64748b;">Manage inputs for analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload Mammography Patch", type=["png", "jpg", "dcm"])
    
    # نمایش وضعیت فایل آپلود شده به سبک HTML شما
    if uploaded_file:
        st.success(f"File loaded: {uploaded_file.name}")
    else:
        st.info("DICOM, PNG, or JPG (Max 5MB)")

    st.markdown("---")
    
    with st.expander("⚙️ Advanced & Model", expanded=True):
        model_type = st.radio(
            "Model Selection",
            ["EfficientNet-B0 (Default)", "ResNet-50"],
            index=0
        )
        st.markdown("---")
        use_clahe = st.checkbox("CLAHE Enhancement", value=True)
        use_noise = st.checkbox("Noise Reduction", value=False)

    # فضای خالی برای هل دادن دکمه به پایین (اختیاری)
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    analyze_btn = st.button("Analyze Image ⚡")
    
    st.markdown("""
    <div style="text-align: center; margin-top: 1rem;">
        <p style="font-size: 0.65rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">System Version</p>
        <p style="font-size: 0.75rem; color: #64748b; font-family: monospace;">Model: v2.1 | Core</p>
    </div>
    """, unsafe_allow_html=True)

# ====================== 4. MAIN LAYOUT ======================
# هدر اصلی
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
        <div style="width: 3rem; height: 3rem; background-color: #ee2b5b; border-radius: 0.75rem; display: flex; align-items: center; justify-content: center; color: white; font-size: 1.5rem; box-shadow: 0 10px 15px -3px rgba(238, 43, 91, 0.3);">
            ✚
        </div>
        <div>
            <h1 style="font-size: 1.875rem; font-weight: 900; line-height: 1.2; margin: 0;">MicroCalc AI</h1>
            <p style="color: #ee2b5b; font-weight: 500; margin: 0;">Breast Cancer Detection System</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_h2:
    # دراپ‌داون زبان (فقط دکوری برای فعلا، یا قابل اتصال به دیکشنری ترجمه)
    st.selectbox("Language", ["English", "Persian (فارسی)"], label_visibility="collapsed")

# ردیف وضعیت (Status Grid)
c1, c2, c3, c4 = st.columns(4)
stats = [
    ("Model Status", "Ready", "🟢"),
    ("Device", "GPU (Cuda)" if torch.cuda.is_available() else "CPU", "💾"),
    ("Inference Time", "~45ms", "⚡"),
    ("Resolution", "224x224", "📐")
]

for col, (label, value, icon) in zip([c1, c2, c3, c4], stats):
    with col:
        st.markdown(f"""
        <div class="stat-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <p class="stat-label">{label}</p>
                <span>{icon}</span>
            </div>
            <p class="stat-value">{value}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ====================== 5. LOGIC & RESULTS ======================
if uploaded_file is not None and analyze_btn:
    image = Image.open(uploaded_file).convert("RGB")
    
    # 1. Prediction Logic
    with st.spinner("Analyzing patterns..."):
        result = predictor.predict(image)
    
    # استخراج مقادیر
    pred_class = result["class"] # 'Malignant' or 'Benign'
    confidence = result["confidence"]
    is_malignant = pred_class == "Malignant"
    is_yellow_flag = result["yellow_flag"]
    
    # تعیین رنگ‌ها بر اساس نتیجه
    theme_color = "#ee2b5b" if is_malignant else "#10b981" # قرمز یا سبز
    
    # ساختار دو ستونه اصلی
    col_left, col_right = st.columns([1.2, 1])
    
    with col_left:
        # --- Diagnostic Card (کد HTML سفارشی) ---
        st.markdown(f"""
        <div class="result-card">
            <div class="result-header">
                <h3 style="margin:0; font-size:1.1rem; font-weight:700; color:#0f172a;">Diagnostic Prediction</h3>
                <span style="font-family:monospace; color:#94a3b8; font-size:0.75rem;">ID: P-2023-AUTO</span>
            </div>
            <div class="result-body">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1.5rem;">
                    <div>
                        <p style="text-transform:uppercase; font-size:0.75rem; font-weight:600; color:#64748b; margin-bottom:0.25rem;">Detected Class</p>
                        <h2 style="font-size:2.5rem; font-weight:900; color:{theme_color}; margin:0;">{pred_class}</h2>
                        <p style="color:#64748b; font-size:0.875rem; margin-top:0.25rem;">
                            The model has detected patterns consistent with {pred_class.lower()} pathology.
                        </p>
                    </div>
                </div>
                
                <div style="display:flex; align-items:end; gap:0.5rem; margin-bottom:0.5rem;">
                    <span style="font-size:2rem; font-weight:700; color:{theme_color};">{confidence:.1%}</span>
                    <span style="font-size:0.875rem; font-weight:500; color:#64748b; padding-bottom:0.5rem;">Confidence</span>
                </div>
                <div style="width:100%; height:0.75rem; background-color:#f1f5f9; border-radius:999px; overflow:hidden;">
                    <div style="width:{confidence*100}%; height:100%; background-color:{theme_color}; box-shadow: 0 0 10px {theme_color}80;"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # --- Yellow Flag Warning ---
        if is_yellow_flag:
            st.warning("⚠️ **Review Required:** Confidence is in the uncertainty zone (45-55%). Clinical correlation recommended.")
        else:
            st.success("✅ **Protocol Validated:** Confidence is within the secure range.")

    with col_right:
        # --- XAI / Grad-CAM Section ---
        st.markdown(f"""
        <div class="result-card" style="height: 100%;">
            <div class="result-header">
                <h3 style="margin:0; font-size:1.1rem; font-weight:700; color:#0f172a;">Explainability (Grad-CAM)</h3>
                <span style="background:#f1f5f9; padding:0.25rem 0.5rem; border-radius:0.25rem; font-size:0.65rem; font-weight:700; color:#64748b;">ATTENTION MAP</span>
            </div>
            <div class="result-body" style="text-align: center;">
        """, unsafe_allow_html=True)
        
        if cam:
            heatmap = generate_gradcam(image, cam)
            st.image(heatmap, use_container_width=True, caption="Model Attention Heatmap")
        else:
            st.info("Grad-CAM not available in Mock Mode or CPU limited mode.")
            st.image(image, use_container_width=True, caption="Original Image")
            
        st.markdown("""
            </div>
        </div>
        """, unsafe_allow_html=True)

elif not analyze_btn:
    # وضعیت اولیه (Waiting)
    st.markdown("""
    <div style="border: 2px dashed #cbd5e1; border-radius: 1rem; padding: 3rem; text-align: center; color: #64748b; margin-top: 2rem;">
        <span style="font-size: 3rem; display: block; margin-bottom: 1rem; opacity: 0.5;">☁️</span>
        <h3 style="font-size: 1.25rem; font-weight: 600; color: #0f172a;">Ready for Analysis</h3>
        <p>Upload a mammography patch from the sidebar and click "Analyze Image".</p>
    </div>
    """, unsafe_allow_html=True)