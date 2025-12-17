# Breast Cancer Microcalcification Classification 🎗️

A deep learning project for classifying breast microcalcifications as **Benign** or **Malignant** using the **CBIS-DDSM** dataset. This project leverages CNN architectures (ResNet, EfficientNet) and advanced preprocessing techniques like CLAHE to improve detection accuracy.

## 🚀 Key Features

*   **Advanced Preprocessing**:
    *   Automatic detection of cropped images vs. ROI masks.
    *   **CLAHE** (Contrast Limited Adaptive Histogram Equalization) for enhancing microcalcification visibility.
*   **Flexible Models**: Support for **ResNet18**, **ResNet34**, and **EfficientNet-B0**.
*   **Robust Training**: Complete pipeline with class weighting, learning rate scheduling, and comprehensive metrics (AUC, F1-Score, Precision, Recall).
*   **Interactive Demo**: Streamlit-based UI for easy model testing and visualization.

## 📂 Dataset

This project uses the **CBIS-DDSM** (Curated Breast Imaging Subset of DDSM) dataset.
*   **Input**: Mammogram patches (grayscale).
*   **Classes**: Benign (0) vs. Malignant (1).
*   **Structure**: Images are organized in folders by patient case. The data loader automatically handles the complex directory structure.

## 🛠️ Installation

### Local Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/breast-cancer-classification.git
    cd breast-cancer-classification
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### ☁️ Google Colab Setup

To run this project in Google Colab, create a new notebook and run the following cell to set up the environment:

```python
!git clone https://github.com/yourusername/breast-cancer-classification.git
%cd breast-cancer-classification
!pip install -r requirements.txt

# Mount Google Drive if your data is stored there
from google.colab import drive
drive.mount('/content/drive')
```

> **Note:** Ensure your dataset is placed in `data/jpeg` and the CSV file in `data/csv/`. You can adjust paths in `core/config.py` if needed.

## 🏃‍♂️ Usage

### 1. Training the Model

Use the `train_script.py` to start training. You can configure the backbone and preprocessing options.

**Train with ResNet34 (recommended):**
```bash
python train_script.py --backbone resnet34 --epochs 10
```

**Train with ResNet18 (smaller, faster):**
```bash
python train_script.py --backbone resnet18 --epochs 10
```

**Options:**
*   `--backbone`: `resnet18`, `resnet34`, `efficientnet_b0`
*   `--epochs`: Number of training epochs (default: 10)
*   `--batch-size`: Batch size (default: 32)

### 2. Running the Demo App

Launch the Streamlit interface to interact with the model:

```bash
streamlit run app/app.py
```

## 📁 Project Structure

```
├── app/
│   └── app.py              # Streamlit Web UI
├── core/
│   ├── config.py           # Configuration settings (Paths, Hyperparams)
│   ├── model.py            # CNN Model definitions (ResNet, EfficientNet)
│   ├── preprocessing.py    # Data loading, CLAHE, and Transforms
│   └── train.py            # Training loop and metrics logic
├── data/
│   ├── csv/                # Metadata CSV files
│   └── jpeg/               # Image directories
├── checkpoints/            # Saved models and training plots
├── train_script.py         # CLI entry point for training
└── requirements.txt        # Python dependencies
```

## 📊 Results

The training pipeline automatically generates:
*   **Learning Curves**: Loss and Accuracy plots saved in `checkpoints/`.
*   **Confusion Matrix**: Visual evaluation of model performance.
*   **Metrics**: Logs Precision, Recall, F1-Score, and AUC-ROC.

---
*Built for the AI Project - Breast Cancer Microcalcification*
