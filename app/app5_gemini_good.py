import streamlit as st
from PIL import Image
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="MicroCalc AI",
    page_icon="⛑️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STATE MANAGEMENT ---
if 'language' not in st.session_state:
    st.session_state['language'] = 'en'
if 'analyzed' not in st.session_state:
    st.session_state['analyzed'] = False

def toggle_language():
    if st.session_state['language'] == 'en':
        st.session_state['language'] = 'fa'
    else:
        st.session_state['language'] = 'en'

# --- TRANSLATIONS ---
t = {
    'en': {
        'dir': 'ltr',
        'font': 'sans-serif',
        'nav_dash': 'Dashboard',
        'nav_about': 'About Us',
        'sb_title': 'Patient Data',
        'sb_upload': 'Upload Mammography Patch',
        'sb_model': 'Advanced & Model',
        'btn_analyze': 'Analyze Image',
        'footer': 'Model: EfficientNet-B0 | v2.1',
        'status_ready': 'Ready',
        'status_device': 'NVIDIA GPU',
        'pred_title': 'Diagnostic Prediction',
        'class_mal': 'Malignant ❗',
        'class_ben': 'Benign ✅',
        'desc_mal': 'The model has detected patterns consistent with malignancy.',
        'desc_ben': 'No malignancy patterns detected.',
        'conf': 'Confidence',
        'proto_val': 'Protocol Validated',
        'proto_desc': 'Quality standards met.',
        'xai_title': 'Explainability Analysis (XAI)',
        'orig_input': 'ORIGINAL INPUT',
        'heatmap': 'ACTIVATION HEATMAP',
        'interp': 'Model Interpretation',
        'interp_desc': 'Red areas indicate regions influencing the model decision.',
        'about_head': 'About MicroCalc Project',
        'about_text': 'This project is developed by the AI Student Association at Kharazmi University. Our goal is to leverage Deep Learning for early breast cancer detection.',
        'team': 'Meet the Team'
    },
    'fa': {
        'dir': 'rtl',
        'font': 'Vazir, Tahoma, sans-serif',
        'nav_dash': 'داشبورد',
        'nav_about': 'درباره ما',
        'sb_title': 'اطلاعات بیمار',
        'sb_upload': 'بارگذاری تصویر ماموگرافی',
        'sb_model': 'تنظیمات مدل و پیشرفته',
        'btn_analyze': 'تحلیل تصویر',
        'footer': 'مدل: EfficientNet-B0 | نسخه 2.1',
        'status_ready': 'آماده',
        'status_device': 'پردازنده گرافیکی',
        'pred_title': 'پیش‌بینی تشخیصی',
        'class_mal': 'بدخیم (Malignant) ❗',
        'class_ben': 'خوش‌خیم (Benign) ✅',
        'desc_mal': 'مدل الگوهای سازگار با بدخیمی را شناسایی کرده است.',
        'desc_ben': 'هیچ الگوی بدخیمی شناسایی نشد.',
        'conf': 'میزان اطمینان',
        'proto_val': 'پروتکل تایید شده',
        'proto_desc': 'استانداردهای کیفی رعایت شده است.',
        'xai_title': 'تحلیل تفسیرپذیری (XAI)',
        'orig_input': 'تصویر ورودی',
        'heatmap': 'نقشه حرارتی (Grad-CAM)',
        'interp': 'تفسیر مدل',
        'interp_desc': 'نواحی قرمز بخش‌هایی هستند که بر تصمیم مدل تاثیر گذاشته‌اند.',
        'about_head': 'درباره پروژه میکروکالک',
        'about_text': 'این پروژه توسط انجمن علمی هوش مصنوعی دانشگاه خوارزمی توسعه یافته است. هدف ما استفاده از یادگیری عمیق برای تشخیص زودهنگام سرطان سینه است.',
        'team': 'تیم توسعه‌دهنده'
    }
}

lang = st.session_state['language']
txt = t[lang]

# --- CUSTOM CSS ---
st.markdown(f"""
<style>
    /* Global Font & Direction */
    .stApp {{
        direction: {txt['dir']};
        font-family: {txt['font']};
        text-align: {'right' if lang == 'fa' else 'left'};
    }}
    
    p, h1, h2, h3, div {{
        text-align: {'right' if lang == 'fa' else 'left'};
    }}

    /* Card Styling */
    div.stMetric, div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"] {{
        background-color: white;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        border: 1px solid #f0f2f6;
    }}
    
    /* Button Styling */
    .stButton > button {{
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        border-radius: 8px;
        height: 50px;
        font-weight: bold;
    }}
    
    /* Malignant Banner */
    .malignant-banner {{
        background-color: #fff0f0;
        border-{'right' if lang == 'fa' else 'left'}: 5px solid #ff4b4b;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
    }}
    
    /* Protocol Card */
    .protocol-card {{
        background-color: #e6f4ea;
        border: 1px solid #c3e6cb;
        padding: 15px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        flex-direction: {'row-reverse' if lang == 'fa' else 'row'};
    }}
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    # Language Toggle
    st.button("English / فارسی", on_click=toggle_language)
    st.markdown("---")
    
    # Navigation
    menu = st.radio("Menu / منو", [txt['nav_dash'], txt['nav_about']])
    st.markdown("---")

    if menu == txt['nav_dash']:
        st.subheader(txt['sb_title'])
        uploaded_file = st.file_uploader(txt['sb_upload'], type=['png', 'jpg', 'dcm'])
        
        with st.expander(txt['sb_model']):
            st.selectbox("Model", ["EfficientNet-B0", "ResNet50"])
            
        st.write("")
        if st.button(txt['btn_analyze']):
            st.session_state['analyzed'] = True
            
        st.markdown("---")
        st.caption(txt['footer'])

# --- PAGE: DASHBOARD ---
if menu == txt['nav_dash']:
    
    # Header
    c1, c2 = st.columns([3, 1])
    with c1:
        st.title("⛑️ MicroCalc AI")
    with c2:
        # Just a visual indicator of language
        st.caption(f"Language: {lang.upper()}")

    # Top Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Status", txt['status_ready'])
    m2.metric("Device", txt['status_device'])
    m3.metric("Latency", "45ms")
    m4.metric("Resolution", "224x224")

    if uploaded_file and st.session_state['analyzed']:
        st.write("")
        st.write("")
        
        # --- PREDICTION ROW (Full Width now) ---
        col_pred, col_proto = st.columns([3, 1])
        
        with col_pred:
            # Prediction Banner
            st.markdown(f"""
            <div class="malignant-banner">
                <div style="color: #666; font-size: 0.9rem;">{txt['pred_title']}</div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 10px;">
                    <div>
                        <div style="color: #ff4b4b; font-size: 2.2rem; font-weight: 800;">{txt['class_mal']}</div>
                        <div style="color: #444;">{txt['desc_mal']}</div>
                    </div>
                    <div style="text-align: center;">
                         <div style="color: #ff4b4b; font-size: 1.8rem; font-weight: 700;">98.2%</div>
                         <div style="font-size: 0.8rem;">{txt['conf']}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_proto:
            # Protocol Card
            st.markdown(f"""
            <div class="protocol-card">
                <div style="font-size: 2rem; margin-{'left' if lang == 'fa' else 'right'}: 15px;">✅</div>
                <div style="text-align: {'right' if lang == 'fa' else 'left'};">
                    <strong>{txt['proto_val']}</strong><br>
                    <span style="font-size: 0.8rem; color: #666;">{txt['proto_desc']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # --- XAI SECTION (Expanded, no sidebar) ---
        st.subheader(f"👁️ {txt['xai_title']}")
        
        # Container for XAI
        with st.container():
            xc1, xc2 = st.columns(2)
            
            with xc1:
                st.caption(txt['orig_input'])
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True)
                
            with xc2:
                st.caption(txt['heatmap'])
                # Placeholder logic: Display same image with tint for demo
                st.image(image, caption="Grad-CAM", use_container_width=True)
        
        # Interpretation Note
        st.info(f"ℹ️ **{txt['interp']}:** {txt['interp_desc']}")

    elif not uploaded_file:
        st.info("👋 " + ("لطفا تصویر را آپلود کنید" if lang == 'fa' else "Please upload an image to start."))

# --- PAGE: ABOUT US ---
elif menu == txt['nav_about']:
    st.title(txt['nav_about'])
    
    st.markdown(f"""
    <div style="background-color: #f8f9fa; padding: 25px; border-radius: 10px; border-{'right' if lang=='fa' else 'left'}: 5px solid #ff4b4b;">
        <h3>{txt['about_head']}</h3>
        <p style="font-size: 1.1rem; line-height: 1.6;">
            {txt['about_text']}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader(txt['team'])
    
    # Team Grid
    tc1, tc2, tc3, tc4 = st.columns(4)
    team_names = ["Member 1", "Member 2", "Member 3", "Member 4"]
    
    for i, col in enumerate([tc1, tc2, tc3, tc4]):
        with col:
            st.image("https://placehold.co/150", use_container_width=True)
            st.markdown(f"**{team_names[i]}**")
            st.caption("AI Researcher")