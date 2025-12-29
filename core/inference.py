import torch
import torch.nn.functional as F
import os
import random
from .config import Config
from .model import BreastCancerModel
from .preprocessing import preprocess_image

class Predictor:
    def __init__(self, model_path=None):
        self.model_path = model_path or Config.MODEL_PATH
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.mock_mode = False
        
        self._load_model()
        
    def _load_model(self):
        if os.path.exists(self.model_path):
            print(f"Loading model from {self.model_path}...")
            try:
                # 1. لود کردن فایل چک‌پوینت
                checkpoint = torch.load(self.model_path, map_location=self.device)
                
                # 2. تشخیص معماری مدل از داخل چک‌پوینت (اگر ذخیره شده باشد)
                # این کار باعث می‌شود اگر مدل با ResNet34 آموزش دیده ولی کانفیگ ResNet18 است، خطا ندهد
                loaded_backbone = None
                if isinstance(checkpoint, dict) and 'backbone' in checkpoint:
                    loaded_backbone = checkpoint['backbone']
                    print(f"Detected backbone in checkpoint: {loaded_backbone}")
                
                # ساخت مدل با معماری درست
                self.model = BreastCancerModel(backbone_name=loaded_backbone)
                
                # 3. استخراج وزن‌ها (State Dict)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    print("Checkpoint contains metadata. Extracting 'model_state_dict'...")
                    state_dict = checkpoint['model_state_dict']
                else:
                    # شاید فایل فقط شامل وزن‌ها باشد (مثل model.pth)
                    state_dict = checkpoint
                
                # 4. بارگذاری وزن‌ها در مدل
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
                print("Model loaded successfully!")
                
            except Exception as e:
                print(f"Error loading model: {e}. Switching to Mock Mode.")
                import traceback
                traceback.print_exc() # چاپ جزئیات خطا برای دیباگ بهتر
                self.mock_mode = True
        else:
            print(f"Model file {self.model_path} not found. Switching to Mock Mode.")
            self.mock_mode = True

    def predict(self, image_path_or_pil):
        """
        Predicts the class of the image.
        """
        if self.mock_mode:
            return self._mock_predict()
            
        # 1. اطمینان از اینکه مدل در حالت ارزیابی است (خاموش کردن Dropout و BatchNorm)
        # این خط حیاتی است برای جلوگیری از نتایج رندوم
        if self.model:
            self.model.eval()

        # 2. پیش‌پردازش
        # نکته: مطمئن شوید در core/preprocessing.py تابع preprocess_image از phase='test' استفاده می‌کند
        tensor = preprocess_image(image_path_or_pil).to(self.device)
        
        # 3. پیش‌بینی
        with torch.no_grad(): # خاموش کردن محاسبه گرادیان
            outputs = self.model(tensor)
            probs = F.softmax(outputs, dim=1)
            confidence, preds = torch.max(probs, 1)
            
        confidence_score = confidence.item()
        predicted_class_idx = preds.item()
        predicted_class = Config.CLASSES[predicted_class_idx]
        
        # Yellow Flag Logic
        yellow_flag = False
        low, high = Config.YELLOW_FLAG_RANGE
        if low <= confidence_score <= high:
            yellow_flag = True
            
        return {
            "class": predicted_class,
            "confidence": confidence_score,
            "yellow_flag": yellow_flag,
            "raw_probabilities": probs.cpu().numpy().tolist()[0],
            "mock": False
        }

    def _mock_predict(self):
        """
        Fake backend for the Demo Team.
        """
        # Simulate a random prediction
        confidence = random.uniform(0.4, 0.99)
        predicted_class_idx = 0 if random.random() < 0.5 else 1
        predicted_class = Config.CLASSES[predicted_class_idx]
        
        # Yellow Flag Logic
        yellow_flag = False
        low, high = Config.YELLOW_FLAG_RANGE
        if low <= confidence <= high:
            yellow_flag = True
            
        return {
            "class": predicted_class,
            "confidence": confidence,
            "yellow_flag": yellow_flag,
            "raw_probabilities": [1-confidence, confidence] if predicted_class_idx == 1 else [confidence, 1-confidence],
            "mock": True
        }
