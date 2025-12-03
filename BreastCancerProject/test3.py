import pandas as pd
import os
import cv2
import numpy as np
from tqdm import tqdm

# تنظیمات
CSV_PATH = "./data/csv/calc_case_description_train_set.csv"
JPEG_ROOT = "./data/jpeg"
COLOR_THRESHOLD = 50  # اگر تعداد رنگ‌ها کمتر از این باشد، یعنی ماسک است

def analyze_full_content():
    print(">>> STARTING FULL CONTENT ANALYSIS (PIXEL BASED) <<<")
    print("-" * 60)

    # 1. لود کردن لیست هدف از CSV
    try:
        df = pd.read_csv(CSV_PATH)
        print(f"Target Samples in CSV: {len(df)}")
    except:
        print("Error: CSV not found.")
        return

    # متغیرهای آمارگیری
    stats = {
        "perfect_folders": 0,    # پوشه‌هایی که دقیقا 1 عکس و 1 ماسک دارند (عالی)
        "only_image": 0,         # پوشه‌هایی که فقط 1 عکس دارند (خوب)
        "double_image": 0,       # پوشه‌هایی که 2 عکس واقعی دارند (گیج‌کننده)
        "only_mask": 0,          # پوشه‌هایی که فقط ماسک دارند (بدرد نخور)
        "missing_folder": 0,     # پوشه پیدا نشد
        "other": 0               # سایر حالت‌ها
    }

    # لیست برای ذخیره خطاهای احتمالی
    problematic_folders = []

    # 2. پیمایش تمام ردیف‌های CSV
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing Images"):
        
        # پیدا کردن مسیر پوشه روی هارد
        csv_path = row['cropped image file path']
        parts = csv_path.split('/')
        
        folder_path = None
        # جستجوی هوشمند پوشه (چون ساختار پوشه‌ها گاهی متفاوت است)
        for part in parts:
            potential = os.path.join(JPEG_ROOT, part)
            if os.path.exists(potential) and os.path.isdir(potential):
                folder_path = potential
                break
        
        if not folder_path:
            stats["missing_folder"] += 1
            continue

        # 3. آنالیز محتویات پوشه پیدا شده
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png'))]
        
        real_images_count = 0
        masks_count = 0
        
        for file_name in files:
            file_full_path = os.path.join(folder_path, file_name)
            
            # خواندن تصویر به صورت سیاه و سفید
            img = cv2.imread(file_full_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                continue # فایل خراب است
                
            # شمارش تعداد رنگ‌های منحصر به فرد
            # ماسک‌ها معمولا 2 رنگ دارند (0 و 255)، ولی بخاطر فشرده‌سازی JPEG ممکن است تا 30-40 هم برود
            unique_colors = len(np.unique(img))
            
            if unique_colors < COLOR_THRESHOLD:
                masks_count += 1
            else:
                real_images_count += 1

        # 4. دسته‌بندی وضعیت این پوشه
        if real_images_count == 1 and masks_count >= 1:
            stats["perfect_folders"] += 1
        elif real_images_count == 1 and masks_count == 0:
            stats["only_image"] += 1
        elif real_images_count > 1:
            stats["double_image"] += 1
            problematic_folders.append(folder_path)
        elif real_images_count == 0 and masks_count > 0:
            stats["only_mask"] += 1
        else:
            stats["other"] += 1

    # 5. گزارش نهایی
    print("\n" + "="*60)
    print(">>> FINAL CONTENT REPORT <<<")
    print("="*60)
    print(f"Total Folders Analyzed: {len(df)}")
    print("-" * 30)
    print(f"✅ Perfect Pairs (1 Image + Mask): {stats['perfect_folders']}  <-- READY FOR TRAINING")
    print(f"🆗 Single Images (1 Image, No Mask): {stats['only_image']}  <-- READY FOR TRAINING")
    print("-" * 30)
    print(f"❌ Double Images (Ambiguous):       {stats['double_image']}")
    print(f"❌ Only Masks (Data Loss):          {stats['only_mask']}")
    print(f"❌ Missing Folders:                 {stats['missing_folder']}")
    print("="*60)

    # تحلیل آماری
    valid_data = stats['perfect_folders'] + stats['only_image']
    print(f"TOTAL VALID TRAINING SAMPLES: {valid_data}")
    
    if valid_data == len(df) - stats['missing_folder']:
        print(">> CONCLUSION: Structure is clean. We can separate based on color count.")
    else:
        print(">> CONCLUSION: We have messy folders. Manual check required for 'Double Images'.")

if __name__ == "__main__":
    analyze_full_content()
print("elahe234")