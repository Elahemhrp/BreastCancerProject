import streamlit as st
from PIL import Image
import time

# --- CONFIGURATION & SETUP ---
st.set_page_config(
    page_title="MicroCalc: Breast Cancer Detection",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STATE MANAGEMENT (Language) ---
if 'language' not in st.session_state:
    st.session_state['language'] = 'en'

def toggle_language():
    if st.session_state['language'] == 'en':
        st.session_state['language'] = 'fa'
    else:
        st.session_state['language'] = 'en'

# --- TRANSLATIONS DICTIONARY ---
t = {
    'en': {
        'dir': 'ltr',
        'font': 'sans-serif',
        'title': 'MicroCalc: Breast Cancer Detection System',
        'nav_dashboard': 'Dashboard',
        'nav_about': 'About Us',
        'sb_title': 'Patient Data',
        'sb_upload': 'Upload Mammography Patch',
        'sb_adv_title': 'Advanced Settings',
        'sb_model_label': 'Select AI Model',
        'sb_model_default': 'Default (EfficientNet-B0)',
        'sb_model_custom': 'Select Custom Model...',
        'btn_analyze': 'Analyze Image',
        'status_ready': 'System Ready',
        'status_device': 'Device: GPU (CUDA)',
        'res_benign': 'Benign',
        'res_malignant': 'Malignant',
        'res_uncertain': '⚠️ Low Confidence Zone. Clinical Review Required.',
        'xai_title': 'Explainability (XAI) Analysis',
        'xai_orig': 'Original Input',
        'xai_grad': 'Grad-CAM Heatmap',
        'xai_desc': 'Red areas indicate regions influencing the model\'s decision.',
        'about_title': 'About MicroCalc',
        'about_desc': 'This project is a University research initiative aimed at leveraging Deep Learning for early breast cancer diagnosis.',
        'team_title': 'Meet the Team',
        'bio_placeholder': 'Computer Science Student | AI Researcher',
        'footer': 'Model: EfficientNet-B0 | v2.1'
    },
    'fa': {
        'dir': 'rtl',
        'font': 'Vazir, Tahoma, sans-serif',
        'title': 'میکروکالک: سیستم تشخیص سرطان سینه',
        'nav_dashboard': 'داشبورد',
        'nav_about': 'درباره ما',
        'sb_title': 'اطلاعات بیمار',
        'sb_upload': 'بارگذاری تصویر ماموگرافی',
        'sb_adv_title': 'تنظیمات پیشرفته',
        'sb_model_label': 'انتخاب مدل هوش مصنوعی',
        'sb_model_default': 'پیش‌فرض (EfficientNet-B0)',
        'sb_model_custom': 'انتخاب مدل سفارشی...',
        'btn_analyze': 'تحلیل تصویر',
        'status_ready': 'سیستم آماده است',
        'status_device': 'پردازنده: GPU',
        'res_benign': 'خوش‌خیم (Benign)',
        'res_malignant': 'بدخیم (Malignant)',
        'res_uncertain': '⚠️ ناحیه عدم قطعیت. نیاز به بررسی بالینی.',
        'xai_title': 'تحلیل و تفسیر پذیری (XAI)',
        'xai_orig': 'تصویر ورودی',
        'xai_grad': 'نقشه حرارتی (Grad-CAM)',
        'xai_desc': 'نواحی قرمز نشان‌دهنده بخش‌های موثر در تصمیم‌گیری مدل هستند.',
        'about_title': 'درباره پروژه',
        'about_desc': 'این پروژه یک طرح دانشگاهی است که با هدف استفاده از یادگیری عمیق برای تشخیص زودهنگام سرطان توسعه یافته است.',
        'team_title': 'تیم توسعه‌دهنده',
        'bio_placeholder': 'دانشجوی علوم کامپیوتر | پژوهشگر هوش مصنوعی',
        'footer': 'مدل: EfficientNet-B0 | نسخه 2.1'
    }
}

lang = st.session_state['language']
txt = t[lang]

# --- CSS FOR RTL/LTR & STYLING ---
# This injects CSS to handle Right-to-Left layout for Persian
st.markdown(f"""
<style>
    /* Main App Container Font & Direction */
    .stApp {{
        direction: {txt['dir']};
        font-family: {txt['font']};
        text-align: {'right' if lang == 'fa' else 'left'};
    }}
    
    /* Adjust headings alignment based on language */
    h1, h2, h3, h4, h5, h6, p, div {{
        text-align: {'right' if lang == 'fa' else 'left'};
    }}

    /* Fix Sidebar Direction separately */
    section[data-testid="stSidebar"] {{
        direction: {txt['dir']};
        text-align: {'right' if lang == 'fa' else 'left'};
    }}
    
    /* Custom Card Styling */
    div.stMetric {{
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }}
    
    /* Center images in columns */
    div[data-testid="stImage"] {{
        display: block;
        margin-left: auto;
        margin-right: auto;
    }}
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    # Language Switcher
    st.button("English / فارسی", on_click=toggle_language, use_container_width=True)
    st.divider()
    
    # Navigation
    page = st.radio("Menu / منو", [txt['nav_dashboard'], txt['nav_about']])
    
    st.divider()
    
    if page == txt['nav_dashboard']:
        st.header(txt['sb_title'])
        uploaded_file = st.file_uploader(txt['sb_upload'], type=['png', 'jpg', 'jpeg'])
        
        with st.expander(txt['sb_adv_title']):
            model_choice = st.selectbox(
                txt['sb_model_label'],
                [txt['sb_model_default'], txt['sb_model_custom']]
            )
            
        analyze_btn = st.button(txt['btn_analyze'], type="primary", use_container_width=True)
        
        st.markdown("---")
        st.caption(txt['footer'])

# --- PAGE: DASHBOARD ---
if page == txt['nav_dashboard']:
    st.title(txt['title'])
    
    # Status Row
    c1, c2, c3 = st.columns(3)
    c1.metric("Status", txt['status_ready'])
    c2.metric("Device", "GPU (CUDA)")
    c3.metric("Latency", "45ms")

    if uploaded_file is not None and analyze_btn:
        with st.spinner('Processing...'):
            time.sleep(1.5) # Simulating processing
            
        # --- PREDICTION SECTION ---
        st.divider()
        
        # MOCK RESULT (Change logic here for real model)
        # Assuming High Confidence Malignant for demo
        prob = 0.92
        prediction_class = "Malignant" 
        
        # Display Result Badge
        col_res, col_space = st.columns([1, 2])
        with col_res:
            if 0.45 <= prob <= 0.55:
                st.warning(f"### {txt['res_uncertain']}")
                st.progress(prob)
            elif prediction_class == "Malignant":
                st.error(f"### {txt['res_malignant']}")
                st.progress(prob)
            else:
                st.success(f"### {txt['res_benign']}")
                st.progress(prob)

        # --- XAI SECTION (Centerpiece) ---
        st.markdown(f"### {txt['xai_title']}")
        st.info(txt['xai_desc'])
        
        # Create a large focused container
        with st.container():
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(txt['xai_orig'])
                # Display uploaded image
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True)
            
            with col2:
                st.subheader(txt['xai_grad'])
                # Placeholder for Heatmap (Using same image with red tint for demo)
                # In real app, replace 'image' with heatmap_array
                st.image(image, caption="Grad-CAM Overlay", use_container_width=True, clamp=True)
            
            # Fullscreen Expander
            with st.expander("🔍 Fullscreen / Detailed View"):
                 st.image(image, caption="High Resolution Heatmap Analysis", use_container_width=True)

    elif not uploaded_file:
        st.info("👋 Please upload a mammography patch from the sidebar to begin.")

# --- PAGE: ABOUT US ---
elif page == txt['nav_about']:
    st.title(txt['about_title'])
    st.markdown(f"""
    <div style="background-color:#f9f9f9; padding:20px; border-radius:10px; border-left: 5px solid #ff4b4b;">
        {txt['about_desc']}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader(txt['team_title'])
    
    # Team Grid
    tc1, tc2, tc3, tc4 = st.columns(4)
    
    team_members = ["Ali", "Sara", "Reza", "Maryam"] # Example Names
    
    for idx, col in enumerate([tc1, tc2, tc3, tc4]):
        with col:
            # Placeholder for profile image
            st.image("https://placehold.co/150", caption=f"Member {idx+1}")
            st.markdown(f"**{team_members[idx]}**")
            st.caption(txt['bio_placeholder'])