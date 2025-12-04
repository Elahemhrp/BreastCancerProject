import sys
import os
import torch
import numpy as np
from PIL import Image

# Add project root to path
sys.path.append(os.path.abspath('.'))

print("1. Testing Imports...")
try:
    from core.config import Config
    from core.preprocessing import apply_clahe, get_transforms, preprocess_image
    from core.model import BreastCancerModel
    from core.inference import Predictor
    print("   [OK] Imports successful.")
except ImportError as e:
    print(f"   [FAIL] Import failed: {e}")
    sys.exit(1)

print("\n2. Testing Preprocessing (CLAHE)...")
try:
    dummy_img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    processed = apply_clahe(dummy_img)
    if processed.shape == (100, 100, 3):
        print("   [OK] CLAHE output shape correct.")
    else:
        print(f"   [FAIL] CLAHE output shape incorrect: {processed.shape}")
except Exception as e:
    print(f"   [FAIL] CLAHE failed: {e}")

print("\n3. Testing Model Initialization...")
try:
    model = BreastCancerModel(backbone_name='resnet18', num_classes=2)
    print("   [OK] Model initialized.")
except Exception as e:
    print(f"   [FAIL] Model initialization failed: {e}")

print("\n4. Testing Inference (Mock Mode)...")
try:
    predictor = Predictor(model_path="non_existent.pth")
    if predictor.mock_mode:
        print("   [OK] Correctly switched to Mock Mode.")
        
        # Create dummy image
        dummy_pil = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        result = predictor.predict(dummy_pil)
        print(f"   [OK] Prediction result: {result}")
        
        if 'yellow_flag' in result:
             print("   [OK] Yellow Flag field present.")
    else:
        print("   [FAIL] Did not switch to Mock Mode.")
except Exception as e:
    print(f"   [FAIL] Inference failed: {e}")

print("\nVerification Complete.")
