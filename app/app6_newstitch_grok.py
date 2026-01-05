import streamlit as st
import base64

# Function to get base64 of an image URL (for embedding if needed, but we'll use URLs directly)
def get_img_as_base64(url):
    import requests
    response = requests.get(url)
    return base64.b64encode(response.content).decode()

# Set page config for wide layout
st.set_page_config(
    page_title="MicroCalc Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to mimic Tailwind styles
st.markdown("""
<style>
    /* Base styles */
    body {
        font-family: 'Inter', sans-serif;
        background-color: #f8f6f6;
    }
    .stApp {
        background-color: #f8f6f6;
    }
    /* Primary color */
    :root {
        --primary: #e92063;
    }
    /* Sidebar styles */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
        padding: 1.5rem;
    }
    /* Buttons */
    button[kind="primary"] {
        background-color: var(--primary);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 0.75rem;
        height: 3rem;
        width: 100%;
    }
    button[kind="secondary"] {
        background-color: white;
        color: var(--primary);
        border: 1px solid var(--primary);
        border-radius: 0.5rem;
    }
    /* Selectbox */
    .stSelectbox > div > div {
        border: 1px solid #d1d5db;
        border-radius: 0.5rem;
        background-color: white;
    }
    /* File uploader */
    .upload-area {
        border: 2px dashed #d1d5db;
        border-radius: 0.75rem;
        background-color: #f9fafb;
        padding: 1.5rem;
        text-align: center;
    }
    /* Diagnosis banner */
    .diagnosis-banner {
        background-color: #ffffff;
        border: 1px solid #fee2e2;
        border-radius: 1rem;
        padding: 1.5rem;
        display: flex;
        flex-direction: column;
        gap: 1.5rem;
    }
    .confidence-bar {
        height: 0.75rem;
        background-color: #f3f4f6;
        border-radius: 9999px;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        background-color: var(--primary);
        width: 94.2%;
    }
    /* Visualization panel */
    .vis-panel {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 1rem;
        overflow: hidden;
    }
    .toolbar {
        background-color: #f9fafb;
        border-bottom: 1px solid #e5e7eb;
        padding: 0.75rem 1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    # Logo and Title
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown('<div style="width: 2.5rem; height: 2.5rem; border-radius: 0.75rem; background-color: var(--primary); display: flex; align-items: center; justify-content: center; color: white; font-size: 1.5rem;">🧬</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<h1 style="font-size: 1.25rem; font-weight: bold;">MicroCalc</h1>', unsafe_allow_html=True)
        st.markdown('<p style="font-size: 0.75rem; color: #6b7280;">AI Diagnostic Tool v2.4</p>', unsafe_allow_html=True)
    
    # Language Toggle
    st.markdown('<div style="display: flex; height: 2.5rem; background-color: #f3f4f6; border-radius: 0.5rem; padding: 0.25rem; margin: 1.5rem 0;">'
                '<label style="flex: 1; display: flex; align-items: center; justify-content: center; background-color: white; border-radius: 0.375rem; font-size: 0.75rem; font-weight: bold;">EN</label>'
                '<label style="flex: 1; display: flex; align-items: center; justify-content: center; border-radius: 0.375rem; font-size: 0.75rem; color: #6b7280;">FA</label>'
                '</div>', unsafe_allow_html=True)
    
    # Navigation
    st.markdown('<a style="display: flex; align-items: center; gap: 0.75rem; padding: 0.75rem 1rem; border-radius: 0.75rem; background-color: var(--primary); opacity: 0.1; color: var(--primary); font-weight: 600; text-decoration: none;">'
                '<span style="font-size: 1.25rem;">📊</span><span style="font-size: 0.875rem;">Dashboard</span></a>', unsafe_allow_html=True)
    st.markdown('<a style="display: flex; align-items: center; gap: 0.75rem; padding: 0.75rem 1rem; border-radius: 0.75rem; color: #4b5563; text-decoration: none;">'
                '<span style="font-size: 1.25rem;">🕒</span><span style="font-size: 0.875rem;">Patient History</span></a>', unsafe_allow_html=True)
    st.markdown('<a style="display: flex; align-items: center; gap: 0.75rem; padding: 0.75rem 1rem; border-radius: 0.75rem; color: #4b5563; text-decoration: none;">'
                '<span style="font-size: 1.25rem;">ℹ️</span><span style="font-size: 0.875rem;">About Us</span></a>', unsafe_allow_html=True)
    
    # Input Image Upload
    st.markdown('<label style="font-size: 0.875rem; font-weight: 600;">Input Image</label>', unsafe_allow_html=True)
    st.markdown('<div class="upload-area">'
                '<div style="width: 3rem; height: 3rem; border-radius: 9999px; background-color: white; display: flex; align-items: center; justify-content: center; margin-bottom: 0.75rem;">'
                '<span style="font-size: 1.5rem; color: var(--primary);">⬆️</span></div>'
                '<p style="font-size: 0.75rem; font-weight: bold;">Upload Mammogram</p>'
                '<p style="font-size: 0.625rem; color: #6b7280;">DICOM or High-Res JPEG</p>'
                '</div>', unsafe_allow_html=True)
    uploader = st.file_uploader("", type=["dicom", "jpg", "jpeg"])
    
    # Model Architecture
    st.markdown('<label style="font-size: 0.875rem; font-weight: 600;">Model Architecture</label>', unsafe_allow_html=True)
    model = st.selectbox("", ["ResNet-50 (Pre-trained)", "Vision Transformer (ViT-B)", "EfficientNet-B7"])
    
    # Analyze Button
    st.button("ANALYZE", type="primary")

# Main Content
# Header
col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
with col1:
    st.markdown('<h2 style="font-size: 1.125rem; font-weight: bold;">Analysis Dashboard</h2>', unsafe_allow_html=True)
with col2:
    st.markdown('<div style="display: flex; align-items: center; gap: 0.5rem; font-size: 0.75rem; color: #6b7280;">'
                '<div style="width: 0.5rem; height: 0.5rem; border-radius: 9999px; background-color: #22c55e;"></div>System Ready</div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div style="display: flex; align-items: center; gap: 0.5rem; font-size: 0.75rem; padding: 0.375rem 0.75rem; background-color: #f3f4f6; border-radius: 9999px;">'
                '🖥️ GPU: NVIDIA A100</div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div style="display: flex; align-items: center; gap: 0.5rem; font-size: 0.75rem;">⚡ Latency: 120ms</div>', unsafe_allow_html=True)
with col5:
    st.image("https://lh3.googleusercontent.com/aida-public/AB6AXuBUkz0_Gr5QkXW6TTLbGKhKqd6mYO3A7zLnYT9bZsTl5LzXXhk90QQvKYKPpd38RDYauQNGidDq2mG4NZA8sl1KUuBZKaV8OoeFKcm_-IAlumSuNVySOyls805at0CFSYyIk647YSKA5mDaT_qctD7fBRgl2TUJHsIsfglWvf0Um7wJZEL8eKFJfB6olfEntZBFCD1R-sy5JW4qPpY5ltxWzxdHilVtxTH1ETLeDzMuO9jDvUpX1BIc6pNmU5mdkRmf04jsgp3ClmrV", width=32)

# Diagnosis Banner
st.markdown('<div class="diagnosis-banner">', unsafe_allow_html=True)
col_banner1, col_banner2, col_banner3 = st.columns([1, 1, 1])
with col_banner1:
    st.markdown('<div style="display: flex; align-items: center; gap: 1rem;">'
                '<div style="width: 3rem; height: 3rem; border-radius: 9999px; background-color: #fee2e2; display: flex; align-items: center; justify-content: center; color: var(--primary);">⚠️</div>'
                '<div><h3 style="font-size: 1.5rem; font-weight: bold;">Malignant Detected</h3>'
                '<p style="font-size: 0.875rem; color: #6b7280;">Based on BI-RADS classification analysis</p></div></div>', unsafe_allow_html=True)
with col_banner2:
    st.markdown('<span style="font-size: 0.75rem; font-weight: 600; color: #6b7280;">CONFIDENCE SCORE</span>', unsafe_allow_html=True)
    st.markdown('<div class="confidence-bar"><div class="confidence-fill"></div></div>', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 0.625rem; text-align: right; color: #9ca3af;">Threshold: 85%</p>', unsafe_allow_html=True)
    st.markdown('<span style="font-size: 1.125rem; font-weight: bold; color: var(--primary);">94.2%</span>', unsafe_allow_html=True)
with col_banner3:
    st.button("Second Opinion", type="secondary")
    st.button("Export Report", type="secondary")
st.markdown('</div>', unsafe_allow_html=True)

# Visualization Panel
st.markdown('<div class="vis-panel">', unsafe_allow_html=True)
# Toolbar
st.markdown('<div class="toolbar">'
            '<div style="display: flex; align-items: center; gap: 1rem;">'
            '<h3 style="font-size: 0.875rem; font-weight: bold; text-transform: uppercase;">XAI VISUALIZATION</h3>'
            '<div style="height: 1rem; width: 1px; background-color: #d1d5db;"></div>'
            '<div style="display: flex; align-items: center; gap: 0.5rem; font-size: 0.75rem; color: #6b7280;">'
            '<div style="width: 0.75rem; height: 0.75rem; border-radius: 9999px; background-color: var(--primary);"></div>High Activation Zone</div></div>'
            '<div style="display: flex; gap: 0.25rem;">'
            '<button style="padding: 0.5rem; border-radius: 0.5rem; color: #6b7280;">+</button>'
            '<button style="padding: 0.5rem; border-radius: 0.5rem; color: #6b7280;">-</button>'
            '<button style="padding: 0.5rem; border-radius: 0.5rem; color: #6b7280;">↺</button>'
            '<div style="height: 1rem; width: 1px; background-color: #d1d5db; margin: 0 0.5rem;"></div>'
            '<button style="padding: 0.5rem; border-radius: 0.5rem; color: #6b7280;">⛶</button>'
            '</div></div>', unsafe_allow_html=True)

# Image Grid
col_img1, col_img2 = st.columns(2)
with col_img1:
    st.markdown('<div style="position: relative; border-radius: 0.75rem; overflow: hidden; border: 1px solid #d1d5db;">'
                '<div style="position: absolute; top: 0.75rem; left: 0.75rem; padding: 0.25rem 0.5rem; border-radius: 0.375rem; background-color: rgba(0,0,0,0.6); color: white; font-size: 0.75rem;">Original Input (L-CC)</div>', unsafe_allow_html=True)
    st.image("https://lh3.googleusercontent.com/aida-public/AB6AXuCkpzNSFRk6Y_weE_8wVStwJ3iMDeHUi0Px9cS6m6f2g_5lNYJGt_FI5L9BehmECg-5yblclrWHGI4ZMyd9ANonq6Vhx_gck0kbKhlVj7Y8vhly7TY262b5a2Xw_KYewjTxhxp-EpKxrBAPgLekHLHeImOP7XTLl1hh1LvP-UHa29wMFJmZjV3eiYUy2REwLIi1_i1qycR81yhsDUzJ9-KPkIeZ06cnrFZDdQfmdhzF14SywLlA5F9QKrZb564TkJRzeRp0xJJDW-S3", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with col_img2:
    st.markdown('<div style="position: relative; border-radius: 0.75rem; overflow: hidden; border: 1px solid #d1d5db;">'
                '<div style="position: absolute; top: 0.75rem; left: 0.75rem; padding: 0.25rem 0.5rem; border-radius: 0.375rem; background-color: rgba(0,0,0,0.6); color: white; font-size: 0.75rem;">Grad-CAM Heatmap</div>', unsafe_allow_html=True)
    st.image("https://lh3.googleusercontent.com/aida-public/AB6AXuCvrHTxTN1bOemlyfmMnV1g0LV4MDhejlxsjnTzPKn1x0m_MLokjMsLkTSv7zBH1Zaf1HGFZ77qFzgDlqHYnXhtAL_kJ3bmkGXn1OLtDZVIUKRKEsz_dFozXFA9wm0pv6K94mQJBWIoYbsql-9PCS6uNXSMEJ4nUzZoFCuowP3Jfe-V8eGPENTpBOQKaFdfd_IaBFBo-TkCT4zaFTYgHz1rhDJKe7DhmE7Sepa5cwZHnnF4C_87E9RjhT9bA6of8nPiS_JHWOXYcs4x", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<footer style="display: flex; justify-content: space-between; align-items: center; font-size: 0.75rem; color: #9ca3af; padding-bottom: 0.5rem;">'
            '<p>© 2024 MicroCalc Systems Inc. All rights reserved.</p>'
            '<div style="display: flex; gap: 1rem;">'
            '<a style="color: #9ca3af; text-decoration: none;">Privacy Policy</a>'
            '<a style="color: #9ca3af; text-decoration: none;">Terms of Service</a>'
            '<span style="color: #d1d5db;">|</span>'
            '<span style="font-style: italic;">Disclaimer: AI results are assistive only.</span>'
            '</div></footer>', unsafe_allow_html=True)