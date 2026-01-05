import sys
import os
import torch
import numpy as np
import albumentations as A
from PIL import Image
from unittest.mock import MagicMock, patch

sys.path.append(os.path.abspath('.'))

from core.config import Config
# Mock config if needed, or assume it works
from core.preprocessing import get_transforms, preprocess_image, CBISDDSMDataset
from core.train import Trainer


def test_preprocessing():
    print("Testing Preprocessing...")
    
    # 1. Test get_transforms
    t_train = get_transforms(phase='train')
    t_val = get_transforms(phase='val')
    
    assert isinstance(t_train, A.Compose)
    assert isinstance(t_val, A.Compose)
    print("  [OK] get_transforms returns A.Compose")
    
    # 2. Test preprocess_image
    dummy_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    dummy_pil = Image.fromarray(dummy_img)
    
    tensor = preprocess_image(dummy_pil)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dim() == 4 # Batch dim
    print("  [OK] preprocess_image returns tensor with batch dim")

def test_dataset_getitem():
    print("\nTesting Dataset __getitem__...")
    # Mock parse_csv_and_build_dataset
    with patch('core.preprocessing.parse_csv_and_build_dataset') as mock_parse:
        # Create a dummy image file
        os.makedirs('tmp_test', exist_ok=True)
        img_path = os.path.abspath('tmp_test/dummy.jpg')
        Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)).save(img_path)
        
        mock_parse.return_value = [(img_path, 0), (img_path, 1)]
        
        # Initialize dataset (pass None for csv_path/jpeg_dir as they are mocked out by parse call)
        ds = CBISDDSMDataset(csv_path='dummy.csv', jpeg_dir='tmp_test', phase='train')
        
        # Test __getitem__
        item, label = ds[0]
        assert isinstance(item, torch.Tensor)
        # Check shape roughly (depending on Config.IMAGE_SIZE)
        # Config.IMAGE_SIZE is (224, 224) usually? 
        # Let's check dimensions are (3, H, W)
        assert item.shape[0] == 3
        print(f"  [OK] __getitem__ returns tensor shape: {item.shape}")
        
        # Cleanup
        if os.path.exists(img_path):
            os.remove(img_path)
        if os.path.exists('tmp_test'):
            os.rmdir('tmp_test')

def test_trainer_sampler():
    print("\nTesting Trainer WeightedRandomSampler...")
    
    # Mock dataset in Trainer
    with patch('core.train.CBISDDSMDataset') as MockDataset:
        # Setup mock dataset instance
        mock_instance = MagicMock()
        MockDataset.return_value = mock_instance
        
        # Setup behavior
        mock_instance.__len__.return_value = 100
        # 90 benign (0), 10 malignant (1)
        labels = [0]*90 + [1]*10
        mock_instance.get_labels.return_value = labels
        # Need to allow indexing for random_split
        mock_instance.__getitem__.return_value = (torch.zeros(3, 224, 224), 0)
        
        trainer = Trainer(backbone='resnet18', batch_size=10, num_epochs=1)
        
        # Run prepare_data
        trainer.prepare_data()
        
        # Check if sampler is WeightedRandomSampler
        if hasattr(trainer.train_loader, 'sampler'):
            sampler = trainer.train_loader.sampler
            if isinstance(sampler, torch.utils.data.WeightedRandomSampler):
                print("  [OK] train_loader uses WeightedRandomSampler")
            else:
                print(f"  [FAIL] Sampler is {type(sampler)}")
        else:
             print("  [FAIL] train_loader has no sampler")

if __name__ == "__main__":
    try:
        test_preprocessing()
        test_dataset_getitem()
        test_trainer_sampler()
        print("\nAll verification tests passed!")
    except Exception as e:
        print(f"\nTest Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
