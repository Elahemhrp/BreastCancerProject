import streamlit as st
from PIL import Image

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="MicroCalc AI: Breast Cancer Detection",
    page_icon="⛑️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR STYLING ---
st.markdown("""
<style>
    /* Main container padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Card Styling */
    div.stMetric, div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"] {
        background-color: white;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        border: 1px solid #f0f2f6;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #e9ecef;
    }
    
    /* Sidebar Button */
    .stButton > button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: #e04343;
    }

    /* Malignant Banner */
    .malignant-banner {
        background-color: #fff0f0;
        border-left: 5px solid #ff4b4b;
        padding: 20px;
        border-radius: 8px;
    }
    .malignant-title {
        color: #ff4b4b;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0;
    }
    .confidence-score {
        color: #ff4b4b;
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Protocol Validated Card */
    .protocol-card {
        background-color: #e6f4ea;
        border: 1px solid #c3e6cb;
        padding: 15px;
        border-radius: 12px;
        display: flex;
        align-items: center;
    }
    
    /* Patch Details Section */
    .patch-details {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #f0f2f6;
    }
    
    /* XAI Section */
    .xai-section {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #f0f2f6;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("👤 Patient Data")
    st.caption("Manage inputs for analysis.")
    
    st.subheader("Mammography Patch")
    uploaded_file = st.file_uploader("Click or drag file (DICOM, PNG, JPG)", type=['png', 'jpg', 'jpeg', 'dcm'])
    
    with st.expander("Advanced & Model"):
        st.selectbox("Select Model", ["EfficientNet-B0 (Default)", "ResNet50", "ViT-Base"])

    st.write("") # Spacer
    st.write("") # Spacer
    
    if st.button("Analyze Image"):
        st.session_state['analyzed'] = True
    else:
        st.session_state['analyzed'] = False

    st.markdown("---")
    st.caption("SYSTEM VERSION\nModel: EfficientNet-B0 | v2.1")

# --- MAIN CONTENT ---

# Header & Language Selector
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("## ⛑️ MicroCalc AI")
    st.caption("Breast Cancer Detection System")
with col2:
    st.selectbox("", ["🌐 English", "🌐 فارسی"], key='lang_select', label_visibility="collapsed")

# Top Metrics Cards
st.write("")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Model Status", "Ready", delta_color="off")
m2.metric("Device", "NVIDIA GPU")
m3.metric("Inference Time", "45ms")
m4.metric("Resolution", "224x224")

# Prediction Section
st.write("")
col_pred, col_protocol = st.columns([2, 1])

with col_pred:
    st.markdown('<div class="malignant-banner">', unsafe_allow_html=True)
    st.caption("DETECTED CLASS")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown('<p class="malignant-title">Malignant ❗</p>', unsafe_allow_html=True)
        st.caption("The model has detected patterns consistent with malignancy.")
    with c2:
        st.markdown('<p class="confidence-score" style="text-align: right;">98.2% <span style="font-size: 1rem; color: #666;">Confidence</span></p>', unsafe_allow_html=True)
        st.progress(98)
    st.markdown('</div>', unsafe_allow_html=True)

with col_protocol:
    st.markdown("""
    <div class="protocol-card">
        <div style="font-size: 1.5rem; margin-right: 10px;">✅</div>
        <div>
            <strong>Protocol Validated</strong><br>
            <span style="font-size: 0.8rem; color: #666;">Quality standards met.</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# XAI & Patch Details Section
st.write("")
col_xai, col_details = st.columns([2, 1])

with col_xai:
    st.markdown('<div class="xai-section">', unsafe_allow_html=True)
    st.subheader("👁️ Explainability Analysis (XAI)")
    st.write("")
    
    xai_c1, xai_c2 = st.columns(2)
    with xai_c1:
        st.caption("ORIGINAL INPUT (DICOM Source)")
        # Placeholder for Original Image
        st.image("https://placehold.co/300x350/e0e0e0/ffffff?text=Original+Mammogram", use_container_width=True)
        
    with xai_c2:
        st.caption("ACTIVATION HEATMAP (Grad-CAM)")
        # Placeholder for Heatmap Image
        st.image("https://placehold.co/300x350/330033/ff0000?text=Heatmap+Overlay", use_container_width=True)
        st.caption("Low 🟦🟨🟧🟥 High")

    st.info("ℹ️ **Model Interpretation:** Red areas indicate regions influencing the model's decision.")
    st.markdown('</div>', unsafe_allow_html=True)

with col_details:
    st.markdown('<div class="patch-details">', unsafe_allow_html=True)
    st.subheader("📋 Patch Details")
    
    st.caption("View Position")
    st.write("**CC (Cranio-Caudal)**")
    st.divider()
    
    st.caption("Laterality")
    st.write("**Left Breast**")
    st.divider()
    
    st.caption("Tissue Density")
    st.write("**Heterogeneously Dense**")
    st.divider()
    
    st.caption("Patient Age Group")
    st.write("**45 - 55 Years**")
    st.markdown('</div>', unsafe_allow_html=True)