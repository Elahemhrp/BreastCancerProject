import streamlit as st
from PIL import Image
import base64

# Page configuration
st.set_page_config(
    page_title="MicroCalc Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Import Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    /* Global styling */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: white;
        border-right: 1px solid #e5e7eb;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 2rem;
    }
    
    /* Primary button styling */
    .stButton > button {
        background-color: #e92063;
        color: white;
        font-weight: 700;
        border: none;
        border-radius: 0.75rem;
        padding: 0.75rem 1.5rem;
        width: 100%;
        font-size: 0.95rem;
        letter-spacing: 0.05em;
        box-shadow: 0 10px 15px -3px rgba(233, 32, 99, 0.25);
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background-color: #d41d58;
        transform: scale(0.98);
    }
    
    /* File uploader styling */
    [data-testid="stFileUploader"] {
        border: 2px dashed #d1d5db;
        border-radius: 0.75rem;
        padding: 2rem;
        background-color: #f9fafb;
        text-align: center;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #e92063;
        background-color: rgba(233, 32, 99, 0.05);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        border-radius: 0.5rem;
        border-color: #d1d5db;
    }
    
    /* Custom header */
    .main-header {
        background-color: white;
        border-bottom: 1px solid #e5e7eb;
        padding: 1rem 2rem;
        margin: -1rem -1rem 2rem -1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    /* Alert banner */
    .alert-banner {
        background: white;
        border: 1px solid rgba(239, 68, 68, 0.2);
        border-left: 4px solid #e92063;
        border-radius: 1rem;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    
    .alert-banner::before {
        content: '';
        position: absolute;
        right: -2rem;
        top: -2rem;
        width: 8rem;
        height: 8rem;
        background: rgba(233, 32, 99, 0.05);
        border-radius: 50%;
        filter: blur(2rem);
    }
    
    /* Progress bar */
    .progress-container {
        background: #f3f4f6;
        border-radius: 9999px;
        height: 0.75rem;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    
    .progress-bar {
        background: #e92063;
        height: 100%;
        border-radius: 9999px;
        transition: width 0.3s;
    }
    
    /* Image container */
    .image-container {
        background: black;
        border-radius: 0.75rem;
        overflow: hidden;
        position: relative;
        border: 1px solid #374151;
    }
    
    .image-label {
        position: absolute;
        top: 0.75rem;
        left: 0.75rem;
        background: rgba(0, 0, 0, 0.6);
        backdrop-filter: blur(8px);
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.375rem;
        font-size: 0.75rem;
        font-weight: 500;
        border: 1px solid rgba(255, 255, 255, 0.1);
        z-index: 10;
    }
    
    /* Logo styling */
    .logo-container {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1.5rem;
    }
    
    .logo-icon {
        width: 2.5rem;
        height: 2.5rem;
        background: #e92063;
        border-radius: 0.75rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        box-shadow: 0 4px 6px rgba(233, 32, 99, 0.3);
    }
    
    .logo-text h1 {
        font-size: 1.25rem;
        font-weight: 800;
        margin: 0;
        color: #1e293b;
    }
    
    .logo-text p {
        font-size: 0.75rem;
        font-weight: 500;
        margin: 0;
        color: #6b7280;
    }
    
    /* Navigation styling */
    .nav-item {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.75rem 1rem;
        border-radius: 0.75rem;
        margin-bottom: 0.25rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .nav-item.active {
        background: rgba(233, 32, 99, 0.1);
        color: #e92063;
    }
    
    .nav-item:hover {
        background: #f9fafb;
    }
    
    /* Status indicators */
    .status-dot {
        width: 0.5rem;
        height: 0.5rem;
        background: #10b981;
        border-radius: 50%;
        animation: pulse 2s infinite;
        display: inline-block;
        margin-right: 0.5rem;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* XAI Visualization header */
    .viz-header {
        background: rgba(249, 250, 251, 0.5);
        border-bottom: 1px solid #e5e7eb;
        padding: 1rem 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    /* Metric styling */
    .metric-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 0.75rem;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    # Logo
    st.markdown("""
    <div class="logo-container">
        <div class="logo-icon">🔬</div>
        <div class="logo-text">
            <h1>MicroCalc</h1>
            <p>AI Diagnostic Tool v2.4</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Language toggle
    col1, col2 = st.columns(2)
    with col1:
        if st.button("EN", key="en", use_container_width=True):
            st.session_state.lang = "EN"
    with col2:
        if st.button("FA", key="fa", use_container_width=True):
            st.session_state.lang = "FA"
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Navigation
    st.markdown("""
    <div class="nav-item active">
        <span>📊</span>
        <span>Dashboard</span>
    </div>
    <div class="nav-item">
        <span>🕐</span>
        <span>Patient History</span>
    </div>
    <div class="nav-item">
        <span>ℹ️</span>
        <span>About Us</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Upload section
    st.markdown("**Input Image**")
    uploaded_file = st.file_uploader(
        "Upload Mammogram",
        type=["dcm", "jpg", "jpeg", "png"],
        label_visibility="collapsed",
        help="DICOM or High-Res JPEG"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Model selection
    st.markdown("**Model Architecture**")
    model = st.selectbox(
        "Model",
        ["ResNet-50 (Pre-trained)", "Vision Transformer (ViT-B)", "EfficientNet-B7"],
        label_visibility="collapsed"
    )
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Analyze button
    if st.button("🔬 ANALYZE", use_container_width=True):
        st.session_state.analyzed = True

# Main content
# Header
st.markdown("""
<div class="main-header">
    <div>
        <h2 style="margin: 0; font-size: 1.125rem; font-weight: 700;">Analysis Dashboard</h2>
    </div>
    <div style="display: flex; gap: 1.5rem; align-items: center; font-size: 0.75rem; color: #6b7280;">
        <div>
            <span class="status-dot"></span>
            <span>System Ready</span>
        </div>
        <div style="display: flex; gap: 0.5rem; align-items: center; background: #f3f4f6; padding: 0.375rem 0.75rem; border-radius: 9999px;">
            <span>💾</span>
            <span style="color: #1e293b; font-weight: 500;">GPU: NVIDIA A100</span>
        </div>
        <div>
            <span>⚡</span>
            <span>Latency: 120ms</span>
        </div>
        <div style="border-left: 1px solid #e5e7eb; padding-left: 1rem; margin-left: 0.5rem;">
            <span style="display: inline-block; width: 2rem; height: 2rem; background: #cbd5e1; border-radius: 50%;">👨‍⚕️</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Diagnosis banner
st.markdown("""
<div class="alert-banner">
    <div style="display: flex; justify-content: space-between; align-items: start; gap: 2rem; position: relative; z-index: 1;">
        <div style="display: flex; gap: 1rem; align-items: start;">
            <div style="width: 3rem; height: 3rem; background: rgba(239, 68, 68, 0.1); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.875rem; flex-shrink: 0;">
                ⚠️
            </div>
            <div>
                <h3 style="font-size: 1.5rem; font-weight: 700; margin: 0 0 0.25rem 0; color: #1e293b;">Malignant Detected</h3>
                <p style="font-size: 0.875rem; color: #6b7280; margin: 0;">Based on BI-RADS classification analysis</p>
            </div>
        </div>
        <div style="min-width: 300px;">
            <div style="display: flex; justify-content: space-between; align-items: end; margin-bottom: 0.5rem;">
                <span style="font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #6b7280;">Confidence Score</span>
                <span style="font-size: 1.125rem; font-weight: 700; color: #e92063;">94.2%</span>
            </div>
            <div class="progress-container">
                <div class="progress-bar" style="width: 94.2%;"></div>
            </div>
            <p style="text-align: right; font-size: 0.625rem; color: #9ca3af; margin: 0.25rem 0 0 0;">Threshold: 85%</p>
        </div>
        <div style="display: flex; gap: 0.75rem; align-items: center;">
            <button style="padding: 0.5rem 1rem; background: white; border: 1px solid #e5e7eb; border-radius: 0.5rem; font-size: 0.875rem; font-weight: 500; cursor: pointer;">Second Opinion</button>
            <button style="padding: 0.5rem 1rem; background: rgba(233, 32, 99, 0.1); color: #e92063; border: 1px solid rgba(233, 32, 99, 0.2); border-radius: 0.5rem; font-size: 0.875rem; font-weight: 700; cursor: pointer;">Export Report</button>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# XAI Visualization section
st.markdown("""
<div style="background: white; border: 1px solid #e5e7eb; border-radius: 1rem; overflow: hidden; margin-bottom: 2rem;">
    <div class="viz-header">
        <div style="display: flex; gap: 1rem; align-items: center;">
            <h3 style="font-size: 0.875rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin: 0;">XAI VISUALIZATION</h3>
            <div style="width: 1px; height: 1rem; background: #d1d5db;"></div>
            <div style="display: flex; gap: 0.5rem; align-items: center; font-size: 0.75rem; color: #6b7280;">
                <span style="width: 0.75rem; height: 0.75rem; background: rgba(233, 32, 99, 0.8); border-radius: 50%; display: inline-block;"></span>
                <span>High Activation Zone</span>
            </div>
        </div>
        <div style="display: flex; gap: 0.25rem;">
            <button style="padding: 0.5rem; background: transparent; border: none; color: #6b7280; cursor: pointer; border-radius: 0.5rem;" title="Zoom In">➕</button>
            <button style="padding: 0.5rem; background: transparent; border: none; color: #6b7280; cursor: pointer; border-radius: 0.5rem;" title="Zoom Out">➖</button>
            <button style="padding: 0.5rem; background: transparent; border: none; color: #6b7280; cursor: pointer; border-radius: 0.5rem;" title="Reset">🔄</button>
            <div style="width: 1px; height: 1rem; background: #d1d5db; margin: 0 0.5rem;"></div>
            <button style="padding: 0.5rem; background: transparent; border: none; color: #6b7280; cursor: pointer; border-radius: 0.5rem;" title="Fullscreen">⛶</button>
        </div>
    </div>
    <div style="padding: 1rem;">
""", unsafe_allow_html=True)

# Image grid
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="image-container" style="aspect-ratio: 1; min-height: 400px;">
        <div class="image-label">Original Input (L-CC)</div>
        <div style="width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; background: #1f2937; color: #6b7280; font-size: 0.875rem;">
            <div style="text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">📊</div>
                <div>Original Mammogram</div>
                <div style="font-size: 0.75rem; opacity: 0.7; margin-top: 0.25rem;">(Upload image to view)</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="image-container" style="aspect-ratio: 1; min-height: 400px;">
        <div class="image-label">Grad-CAM Heatmap</div>
        <div style="width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; background: linear-gradient(135deg, rgba(233, 32, 99, 0.2), rgba(233, 32, 99, 0.05)); color: #6b7280; font-size: 0.875rem;">
            <div style="text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">🔥</div>
                <div>AI Heatmap Overlay</div>
                <div style="font-size: 0.75rem; opacity: 0.7; margin-top: 0.25rem;">(Generated after analysis)</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div></div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; padding-top: 2rem; font-size: 0.75rem; color: #9ca3af; border-top: 1px solid #e5e7eb; margin-top: 2rem;">
    <p style="margin: 0;">© 2024 MicroCalc Systems Inc. All rights reserved.</p>
    <div style="display: flex; gap: 1rem; align-items: center;">
        <a href="#" style="color: #9ca3af; text-decoration: none;">Privacy Policy</a>
        <a href="#" style="color: #9ca3af; text-decoration: none;">Terms of Service</a>
        <span style="color: #d1d5db;">|</span>
        <span style="font-style: italic;">Disclaimer: AI results are assistive only.</span>
    </div>
</div>
""", unsafe_allow_html=True)