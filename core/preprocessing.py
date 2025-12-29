import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Optional, List, Tuple
import os
import glob
import logging

from .config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def apply_clahe(image_np, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to a numpy image.
    Args:
        image_np: Numpy array of the image (H, W, C) or (H, W).
        clip_limit: Threshold for contrast limiting.
        tile_grid_size: Size of grid for histogram equalization.
    Returns:
        Processed image as numpy array (H, W, C) in RGB format.
    """
    # Convert to grayscale if RGB
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np.copy()
        
    # Ensure uint8
    if gray.dtype != np.uint8:
        gray = (gray * 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    equalized = clahe.apply(gray)
    
    # Convert back to RGB (3 channels) for model compatibility
    equalized_rgb = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)
    return equalized_rgb



def get_transforms(phase="train"):
    """
    Get Albumentations transforms for training or inference.
    Args:
        phase: 'train' or 'val'/'test'.
    Returns:
        A.Compose object of transforms.
    """
    # Base transforms (CLAHE + Normalize + ToTensor)
    base_transforms = [
        A.CLAHE(p=1.0, clip_limit=(2.0, 2.0), tile_grid_size=(8, 8)),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]

    if phase == "train":
        return A.Compose([
             # Augmentations
             A.Resize(height=Config.IMAGE_SIZE[0], width=Config.IMAGE_SIZE[1]), # Ensure size
             *base_transforms[:1], # CLAHE first
             A.HorizontalFlip(p=0.5), 
             A.VerticalFlip(p=0.5),
             A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.7),
             A.GaussNoise(p=0.3),
             # A.CoarseDropout(max_holes=6, max_height=8, max_width=8, min_holes=1, p=0.3),
             *base_transforms[1:], # Normalize and ToTensor
        ])
    
    else:
        # Validation/Test
        return A.Compose([
            A.Resize(height=Config.IMAGE_SIZE[0], width=Config.IMAGE_SIZE[1]),
        ] + base_transforms)


def preprocess_image(image_path_or_pil):
    """
    Full preprocessing pipeline: Load -> Transform (Resize, CLAHE, Normalize, ToTensor) -> Batch Dimension.
    
    Args:
        image_path_or_pil: Path to image or PIL Image object.
    
    Returns:
        Preprocessed tensor with batch dimension.
    """
    if isinstance(image_path_or_pil, str):
        image = Image.open(image_path_or_pil).convert("RGB")
    else:
        image = image_path_or_pil.convert("RGB")
        
    image_np = np.array(image)
    
    # Get transforms (Test phase)
    transform = get_transforms(phase="test")
    
    # Apply Transforms (Albumentations expects image=np_array)
    augmented = transform(image=image_np)
    tensor = augmented['image']
    
    # Add batch dimension
    return tensor.unsqueeze(0)


# ============================================================================
# CBIS-DDSM Dataset Functions
# ============================================================================

def extract_folder_name(cropped_image_path: str) -> Optional[str]:
    """
    Extract folder name from the CSV's 'cropped image file path'.
    
    Example input:
      "Calc-Training_P_00005_RIGHT_CC_1/1.3.6.../1.3.6.../000001.dcm"
    
    Logic: Split by '/' and take second-to-last element.
    
    Returns: 
        Folder name like "1.3.6.1.4.1.9590.100.1.2.393344..." 
        or None if parsing fails.
    """
    if not cropped_image_path or pd.isna(cropped_image_path):
        return None
    
    # Clean the path (remove whitespace, newlines)
    path = str(cropped_image_path).strip()
    
    # Split by '/'
    parts = path.split('/')
    
    # We need at least 3 parts: [case_folder, series_folder, folder_name, filename]
    if len(parts) < 3:
        return None
    
    # Get second-to-last element (the folder containing the images)
    folder_name = parts[-2].strip()
    
    return folder_name if folder_name else None


def count_unique_colors(image_path: str, sample_size: int = 100) -> int:
    """
    Count unique intensity values in a grayscale image.
    Used to distinguish ROI masks (<15 colors) from cropped images (>15 colors).
    
    Args:
        image_path: Path to the image file.
        sample_size: Size of center patch to sample for faster processing.
        
    Returns:
        Number of unique color/intensity values.
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0
        
        # Sample center patch for faster processing
        h, w = img.shape
        if h > sample_size and w > sample_size:
            y1 = (h - sample_size) // 2
            x1 = (w - sample_size) // 2
            patch = img[y1:y1+sample_size, x1:x1+sample_size]
        else:
            patch = img
            
        unique_colors = len(np.unique(patch))
        return unique_colors
    except Exception as e:
        logger.warning(f"Error counting colors in {image_path}: {e}")
        return 0


def select_cropped_image(folder_path: str, threshold: int = None) -> Optional[str]:
    """
    Given a folder with 2 images, select the one that is the cropped image
    (has more unique colors than threshold).
    
    Args:
        folder_path: Path to the folder containing images.
        threshold: Color count threshold. Images with more colors are cropped images.
                   Defaults to Config.COLOR_THRESHOLD.
    
    Returns:
        Path to the cropped image, or None if selection fails.
    """
    if threshold is None:
        threshold = Config.COLOR_THRESHOLD
        
    if not os.path.exists(folder_path):
        return None
    
    # Find all image files in the folder
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    if len(image_files) == 0:
        logger.warning(f"No images found in {folder_path}")
        return None
    
    if len(image_files) == 1:
        # Only one image, return it
        return image_files[0]
    
    if len(image_files) != 2:
        logger.warning(f"Expected 2 images in {folder_path}, found {len(image_files)}")
        # Still try to find the best one
    
    # Count colors for each image and select the one with more colors
    best_image = None
    max_colors = 0
    
    for img_path in image_files:
        colors = count_unique_colors(img_path)
        if colors > max_colors:
            max_colors = colors
            best_image = img_path
    
    # Verify the selected image has enough colors to be a cropped image
    if max_colors <= threshold:
        logger.warning(f"Best image in {folder_path} has only {max_colors} colors, may be a mask")
    
    return best_image


def parse_csv_and_build_dataset(csv_path: str, jpeg_dir: str) -> List[Tuple[str, int]]:
    """
    Parse the CSV metadata, extract folder names, locate images, and return
    a list of (image_path, label) tuples.
    
    Args:
        csv_path: Path to the CSV metadata file.
        jpeg_dir: Path to the directory containing jpeg folders.
        
    Returns:
        List of (image_path, label) tuples where label is 0 (benign) or 1 (malignant).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    if not os.path.exists(jpeg_dir):
        raise FileNotFoundError(f"JPEG directory not found: {jpeg_dir}")
    
    df = pd.read_csv(csv_path)
    
    # Required columns
    path_col = 'cropped image file path'
    pathology_col = 'pathology'
    
    if path_col not in df.columns:
        raise ValueError(f"Column '{path_col}' not found in CSV")
    if pathology_col not in df.columns:
        raise ValueError(f"Column '{pathology_col}' not found in CSV")
    
    dataset = []
    skipped = 0
    
    for idx, row in df.iterrows():
        # Extract folder name
        folder_name = extract_folder_name(row[path_col])
        if folder_name is None:
            skipped += 1
            continue
        
        # Build folder path
        folder_path = os.path.join(jpeg_dir, folder_name)
        
        # Select the cropped image
        image_path = select_cropped_image(folder_path)
        if image_path is None:
            skipped += 1
            continue
        
        # Determine label from pathology
        pathology = str(row[pathology_col]).upper().strip()
        if 'MALIGNANT' in pathology:
            label = 1
        else:  # BENIGN or BENIGN_WITHOUT_CALLBACK
            label = 0
        
        dataset.append((image_path, label))
    
    logger.info(f"Loaded {len(dataset)} samples, skipped {skipped}")
    return dataset


class CBISDDSMDataset(Dataset):
    """
    PyTorch Dataset for CBIS-DDSM calcification images.
    
    Args:
        csv_path: Path to CSV metadata file. Defaults to Config.TRAIN_DESC_CSV.
        jpeg_dir: Path to jpeg images directory. Defaults to Config.JPEG_DIR.
        phase: 'train' or 'val'/'test' (affects augmentation).
    
    Pipeline:
        1. Parse CSV metadata
        2. Locate correct folders in jpeg directory
        3. Select cropped images (not ROI masks) based on color count
        4. Apply transforms (CLAHE, resize, normalize, augmentation)
        5. Return (tensor, label) pairs
    """
    
    def __init__(
        self, 
        csv_path: str = None, 
        jpeg_dir: str = None, 
        phase: str = 'train', 
    ):
        self.csv_path = csv_path or Config.TRAIN_DESC_CSV
        self.jpeg_dir = jpeg_dir or Config.JPEG_DIR
        self.phase = phase
        
        # Parse CSV and build list of (image_path, label)
        self.samples = parse_csv_and_build_dataset(self.csv_path, self.jpeg_dir)
        
        # Get transforms
        self.transform = get_transforms(phase=self.phase)
        
        logger.info(f"CBISDDSMDataset initialized: {len(self.samples)} samples, "
                    f"phase={self.phase}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        
        # Apply transforms (Augmentation + CLAHE + Normalize + ToTensor)
        augmented = self.transform(image=image_np)
        tensor = augmented['image']
        
        return tensor, label
    
    def get_labels(self) -> List[int]:
        """Get all labels for computing class weights."""
        return [label for _, label in self.samples]
    
    def get_class_distribution(self) -> dict:
        """Get class distribution statistics."""
        labels = self.get_labels()
        benign = labels.count(0)
        malignant = labels.count(1)
        return {
            'benign': benign,
            'malignant': malignant,
            'total': len(labels),
            'benign_ratio': benign / len(labels) if labels else 0,
            'malignant_ratio': malignant / len(labels) if labels else 0
        }
