import os

class Config:
    # Data Paths
    DATA_DIR = "data"
    JPEG_DIR = os.path.join(DATA_DIR, "jpeg")
    CSV_DIR = os.path.join(DATA_DIR, "csv")
    TRAIN_DESC_CSV = os.path.join(CSV_DIR, "calc_case_description_train_set.csv")
    TEST_DESC_CSV = os.path.join(CSV_DIR, "calc_case_description_test_set.csv")
    
    # Legacy paths (kept for compatibility)
    TRAIN_CSV = os.path.join(DATA_DIR, "calc_train_paths.csv")
    TEST_CSV = os.path.join(DATA_DIR, "calc_test_paths.csv")
    
    # Image Selection (for distinguishing ROI mask vs cropped image)
    COLOR_THRESHOLD = 15  # Images with more unique colors than this are cropped images
    
    # Model Settings
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 64
    NUM_CLASSES = 2  # Benign, Malignant
    BACKBONE = "resnet34"  # Options: resnet18, resnet34, efficientnet_b0
    
    # Training Settings
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 10
    TRAIN_VAL_SPLIT = 0.8  # 80% train, 20% validation
    
    # Paths
    SAVE_DIR = "checkpoints"
    MODEL_PATH = os.path.join(SAVE_DIR, "model.pth")
    
    # Inference Settings
    YELLOW_FLAG_RANGE = (0.45, 0.55)
    
    # Class Names
    CLASSES = ["Benign", "Malignant"]
