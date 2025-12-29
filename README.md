
# 🎗️ Breast Cancer Detection System (Full-Stack)

## 📌 Project Overview

This project is a comprehensive **Full-Stack Breast Cancer Detection System** utilizing Deep Learning on mammogram images (**CBIS-DDSM** dataset).

Unlike traditional monolithic scripts, this project features a modern **Client-Server architecture**:

- **Backend:** A high-performance **FastAPI** server that handles AI inference, preprocessing, and Explainable AI (XAI) generation.
- **Frontend:** A responsive, interactive web dashboard built with **React**, **Vite**, and **Shadcn UI** for a seamless user experience.

The system classifies microcalcifications/masses as **Benign** or **Malignant** and provides **Grad-CAM heatmaps** to visualize the model's focus area.

---

## 🚀 Key Features

### 🧠 AI & Backend Core
- **Smart Preprocessing Pipeline:**
  - **Heuristic Data Selection:** Automatically distinguishes between "cropped tissue images" and "binary ROI masks" using color histogram analysis (>15 unique colors).
  - **CLAHE Enhancement:** Applies *Contrast Limited Adaptive Histogram Equalization* to reveal hidden details in dense breast tissue.
- **Flexible Models:** Supports **ResNet18**, **ResNet34**, and **EfficientNet** backbones via Transfer Learning.
- **Explainable AI (XAI):** Generates **Grad-CAM heatmaps** to make the "black box" decisions interpretable for medical professionals.

### 💻 Frontend & UI
- **Modern Dashboard:** Built with **React** and **Tailwind CSS**.
- **Interactive Visualization:** Real-time upload, confidence score display, and toggleable heatmap overlays.
- **Decoupled Architecture:** Communicates with the backend via a RESTful API (`/predict`).

---

## 📂 Project Structure

```text
root/
│
├── api.py                   # 🧠 Backend: FastAPI entry point & endpoints
├── train_script.py          # 🏋️ Training: CLI script for model training
├── requirements.txt         # 📋 Dependencies (Python)
│
├── core/                    # ⚙️ Core Logic
│   ├── config.py            # Configuration & Constants
│   ├── model.py             # PyTorch Model Definitions
│   ├── preprocessing.py     # CLAHE, Transforms, & Data Loading
│   └── inference.py         # Prediction Logic & Grad-CAM Wrapper
│
├── frontend/                # 🎨 Frontend: React Application
│   ├── src/                 # Source code (Components, Pages, Hooks)
│   ├── package.json         # JS Dependencies
│   └── ...
│
├── checkpoints/             # 💾 Saved Models (.pth) & Training Graphs
└── data/                    # 📁 Dataset (Images & CSVs)

```

---

## 🛠️ Installation & Run Guide

To run the full system, you need to set up two separate terminals: one for the **Backend (Python)** and one for the **Frontend (React)**.

### 1️⃣ Backend Setup (Terminal 1)

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Start the FastAPI server
uvicorn api:app --reload --host 0.0.0.0 --port 8000

```

✅ *The API will be live at `http://localhost:8000` (Docs at `/docs`).*

### 2️⃣ Frontend Setup (Terminal 2)

```bash
# 1. Navigate to the frontend directory
cd frontend

# 2. Install JavaScript packages
npm install
# Note: If you use bun, run: bun install

# 3. Start the development server
npm run dev

```

✅ *The web app will launch at `http://localhost:5173`.*

---

## 🏋️ Training the Model (Optional)

If you want to retrain the model from scratch instead of using the pre-trained weights, use the `train_script.py`:

```bash
# Train with ResNet34 and CLAHE enabled
python train_script.py --backbone resnet34 --use-clahe --epochs 10

```

**Training Arguments:**

* `--backbone`: `resnet18`, `resnet34`, `efficientnet_b0`
* `--no-clahe`: Disable CLAHE preprocessing.
* `--batch-size`: Set batch size (default: 32).

---

## 📊 Methodology

1. **Input:** User uploads a mammogram patch via the React UI.
2. **Validation:** Backend checks if the image is a valid tissue scan (not a mask) using `count_unique_colors`.
3. **Enhancement:** Image is processed with **CLAHE** (ClipLimit=2.0).
4. **Inference:** The model predicts the probability of malignancy.
5. **Explanation:** **Grad-CAM** computes gradients for the predicted class to generate a heatmap.
6. **Output:** JSON response containing Class, Confidence Score, and Base64 Heatmap is sent to the frontend.

```

```
