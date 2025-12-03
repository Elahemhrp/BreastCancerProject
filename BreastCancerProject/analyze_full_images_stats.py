import pandas as pd
import os
import numpy as np
from PIL import Image # استفاده از PIL برای خواندن سریع ابعاد بدون لود کردن کل عکس
from tqdm import tqdm

# --- تنظیمات ---
CSV_PATH = "./data/csv/calc_case_description_train_set.csv"
JPEG_ROOT = "./data/jpeg"

def analyze_full_mammograms():
    print(">>> FULL MAMMOGRAM STATISTICS (HDD SCAN) <<<")
    print("-" * 60)

    # 1. خواندن CSV
    try:
        df = pd.read_csv(CSV_PATH)
        print(f"Target Rows in CSV: {len(df)}")
    except:
        print("Error: CSV not found.")
        return

    # متغیرهای آمارگیری
    found_count = 0
    missing_count = 0
    widths = []
    heights = []
    
    # لیست برای پیدا کردن فایل‌ها
    # ما باید ستون image file path را چک کنیم
    
    print("Scanning specific full-image paths...")
    
    for index, row in tqdm(df.iterrows(), total=len(df)):
        csv_full_path = row['image file path']
        
        # استخراج UID پوشه از آدرس طولانی
        parts = csv_full_path.split('/')
        
        folder_path = None
        # جستجو برای پیدا کردن پوشه متناظر روی هارد
        for part in parts:
            potential = os.path.join(JPEG_ROOT, part)
            if os.path.exists(potential) and os.path.isdir(potential):
                folder_path = potential
                break
        
        if folder_path:
            # پیدا کردن فایل عکس داخل پوشه
            files = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]
            
            if len(files) > 0:
                # معمولا فقط یک عکس کامل در این پوشه است
                img_path = os.path.join(folder_path, files[0])
                
                try:
                    # خواندن ابعاد عکس (Lazy Load)
                    with Image.open(img_path) as img:
                        w, h = img.size
                        widths.append(w)
                        heights.append(h)
                        found_count += 1
                except Exception as e:
                    print(f"Error reading image {img_path}: {e}")
            else:
                missing_count += 1 # پوشه هست ولی خالیه
        else:
            missing_count += 1 # اصلا پوشه پیدا نشد

    # --- گزارش نهایی ---
    print("\n" + "="*60)
    print(">>> FULL IMAGE STATS REPORT <<<")
    print("="*60)
    
    print(f"1. QUANTITY (تعداد):")
    print(f"   - Total Full Images Expected: {len(df)}")
    print(f"   - Total Full Images Found:    {found_count}")
    print(f"   - Missing Files/Folders:      {missing_count}")
    
    if found_count > 0:
        avg_w = int(np.mean(widths))
        avg_h = int(np.mean(heights))
        min_w, min_h = np.min(widths), np.min(heights)
        max_w, max_h = np.max(widths), np.max(heights)
        
        print("-" * 40)
        print(f"2. DIMENSIONS (ابعاد - پیکسل):")
        print(f"   - Average Size:  {avg_w} x {avg_h} pixels")
        print(f"   - Smallest Img:  {min_w} x {min_h}")
        print(f"   - Largest Img:   {max_w} x {max_h}")
        
        # محاسبه مگاپیکسل میانگین
        mp = (avg_w * avg_h) / 1_000_000
        print(f"   - Average Megapixels: {mp:.2f} MP")
        
        print("-" * 40)
        print("3. ANALYSIS FOR HARDWARE:")
        print(f"   If you use these images, you must resize {avg_w}x{avg_h} -> 224x224.")
        resize_ratio = 224 / avg_w
        print(f"   Resize Ratio: {resize_ratio:.4f} (Image becomes {resize_ratio*100:.2f}% of original size)")
        print("   >> RESULT: Microcalcifications will vanish completely.")

    else:
        print("\nNo full images were found to analyze.")

if __name__ == "__main__":
    analyze_full_mammograms()