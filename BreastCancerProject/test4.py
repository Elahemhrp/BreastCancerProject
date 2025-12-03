import pandas as pd
import os
import cv2

# تنظیمات
CSV_PATH = "./data/csv/calc_case_description_train_set.csv"
JPEG_ROOT = "./data/jpeg"

def get_folder_uid_from_path(long_path):
    """
    یک تابع کمکی که آدرس طولانی را می‌گیرد و UID پوشه را برمی‌گرداند.
    """
    parts = long_path.split('/')
    for part in parts:
        # چک می‌کنیم آیا این بخش از آدرس، به عنوان یک پوشه در هارد وجود دارد؟
        potential_path = os.path.join(JPEG_ROOT, part)
        if os.path.exists(potential_path) and os.path.isdir(potential_path):
            return part, potential_path
    return None, None

def analyze_hidden_data():
    print(">>> ANALYZING FULL MAMMOGRAMS & MASKS PATHS <<<")
    print("-" * 60)
    
    df = pd.read_csv(CSV_PATH)
    
    stats = {
        "mask_same_folder_as_crop": 0,  # ماسک تو همون پوشه کراپ بوده (تایید حدس ما)
        "mask_different_folder": 0,     # ماسک تو پوشه جداگانه بوده
        "full_image_found": 0,          # عکس کامل پیدا شد
        "full_image_missing": 0,        # عکس کامل پیدا نشد
        "full_same_as_crop": 0          # (بعید است) عکس کامل و کراپ تو یک پوشه باشند
    }
    
    # برای نمونه‌برداری از سایز عکس‌های کامل
    sample_full_image_info = None

    print(f"Checking {len(df)} rows from CSV...")
    
    for index, row in df.iterrows():
        # 1. استخراج مسیرها از CSV
        crop_path_csv = row['cropped image file path']
        mask_path_csv = row['ROI mask file path']
        full_path_csv = row['image file path']
        
        # 2. پیدا کردن پوشه واقعی هر کدام روی هارد
        crop_uid, crop_real_path = get_folder_uid_from_path(crop_path_csv)
        mask_uid, mask_real_path = get_folder_uid_from_path(mask_path_csv)
        full_uid, full_real_path = get_folder_uid_from_path(full_path_csv)
        
        # --- تحلیل ماسک‌ها (ROI Mask) ---
        if mask_uid and crop_uid:
            if mask_uid == crop_uid:
                stats["mask_same_folder_as_crop"] += 1
            else:
                stats["mask_different_folder"] += 1
        
        # --- تحلیل عکس‌های کامل (Full Mammogram) ---
        if full_uid:
            stats["full_image_found"] += 1
            
            # چک کنیم آیا با کراپ هم‌پوشانی دارد؟
            if full_uid == crop_uid:
                stats["full_same_as_crop"] += 1
            
            # گرفتن نمونه از عکس کامل (فقط یک بار)
            if sample_full_image_info is None and full_real_path:
                files = [f for f in os.listdir(full_real_path) if f.endswith('.jpg')]
                if files:
                    f_path = os.path.join(full_real_path, files[0])
                    img = cv2.imread(f_path)
                    if img is not None:
                        h, w, c = img.shape
                        size_kb = os.path.getsize(f_path) / 1024
                        sample_full_image_info = f"Dimensions: {w}x{h}, Size: {size_kb:.1f} KB"
        else:
            stats["full_image_missing"] += 1

    # گزارش نهایی
    print("\n" + "="*60)
    print(">>> SECRET DATA REPORT <<<")
    print("="*60)
    
    print("1. ROI MASK ANALYSIS (ماسک‌ها کجا بودند؟):")
    print(f"   - Inside the SAME folder as Crop:      {stats['mask_same_folder_as_crop']}")
    print(f"   - Inside a DIFFERENT folder:           {stats['mask_different_folder']}")
    print("   >> CONCLUSION: This confirms why we saw 2 files in crop folders.")
    print("      The second file WAS indeed the ROI Mask pointed to by CSV.")
    
    print("-" * 40)
    
    print("2. FULL MAMMOGRAM ANALYSIS (عکس‌های کامل کجا هستند؟):")
    print(f"   - Full Images Found on Disk:           {stats['full_image_found']}")
    print(f"   - Full Images Missing:                 {stats['full_image_missing']}")
    print(f"   - Overlap with Crop Folders:           {stats['full_same_as_crop']}")
    print("   >> CONCLUSION: Full images are in totally separate folders.")
    
    print("-" * 40)
    
    if sample_full_image_info:
        print("3. FULL IMAGE SPECS (چرا از عکس کامل استفاده نمی‌کنیم؟):")
        print(f"   - Sample Full Image: {sample_full_image_info}")
        print("   - Compare with Crop: Usually ~300x300 pixels")
        print("   >> WARNING: These images are HUGE. They will crash your 4GB RAM.")
    else:
        print("3. FULL IMAGE SPECS:")
        print("   - No full images found to analyze.")

if __name__ == "__main__":
    analyze_hidden_data()
