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
                self.model = BreastCancerModel()
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
            except Exception as e:
                print(f"Error loading model: {e}. Switching to Mock Mode.")
                self.mock_mode = True
        else:
            print(f"Model file {self.model_path} not found. Switching to Mock Mode.")
            self.mock_mode = True

    def predict(self, image_path_or_pil):
        """
        Predicts the class of the image.
        Returns a dictionary with:
            - class: Predicted class name
            - confidence: Confidence score (0-1)
            - yellow_flag: Boolean, true if confidence is uncertain
            - raw_probabilities: List of probabilities
            - mock: Boolean, true if mock mode was used
        """
        if self.mock_mode:
            return self._mock_predict()
            
        # Real Prediction
        tensor = preprocess_image(image_path_or_pil).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = F.softmax(outputs, dim=1)
            confidence, preds = torch.max(probs, 1)
            
        confidence_score = confidence.item()
        predicted_class_idx = preds.item()
        predicted_class = Config.CLASSES[predicted_class_idx]
        
        # Yellow Flag Logic
        yellow_flag = False
        low, high = Config.YELLOW_FLAG_RANGE
        # Check if the max probability is within the uncertain range
        # Note: If binary classification, uncertainty is around 0.5.
        # If confidence is 0.51, it's uncertain.
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
