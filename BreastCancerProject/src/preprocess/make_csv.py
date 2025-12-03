import os
import pandas as pd
import cv2
import numpy as np

# ================= تنظیمات مسیر =================
BASE_DIR = os.getcwd()
# توجه: این کد را برای Test هم باید جداگانه اجرا کنی (با تغییر CSV_PATH)
CSV_PATH = os.path.join(BASE_DIR, "data/csv/calc_case_description_train_set.csv")
JPEG_ROOT = os.path.join(BASE_DIR, "data/jpeg")
OUTPUT_CSV = "calc_train_paths.csv"

# ================= توابع فیلتر =================
def is_valid_patch(img_path):
    """
    این تابع چک می‌کند که عکس:
    1. ماسک نباشد.
    2. عکس کامل (Full Mammogram) نباشد.
    """
    try:
        # خواندن عکس (Grayscale)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: return False
        
        h, w = img.shape
        unique_vals = len(np.unique(img))
        
        # 1. فیلتر ماسک (ROI)
        # اگر کمتر از 30 رنگ داشت، ماسک است.
        if unique_vals < 30:
            return False
            
        # 2. فیلتر عکس خیلی بزرگ (Full Mammogram)
        # پچ‌های استاندارد معمولا زیر 1500 پیکسل هستند.
        # اگر خیلی بزرگ بود، یعنی عکس کامل سینه است که به درد ما نمی‌خورد.
        if h > 2000 or w > 2000:
            return False
            
        # اگر از هر دو فیلتر رد شد، یعنی پچ مناسب است
        return True
        
    except Exception:
        return False

# ================= بدنه اصلی =================
print("🚀 Starting FINAL Dataset Creation (Bulletproof Mode)...")
df = pd.read_csv(CSV_PATH)

data_rows = []
stats = {
    "saved": 0,
    "skipped_mask_only": 0,
    "skipped_full_only": 0,
    "skipped_empty": 0
}

for idx, row in df.iterrows():
    # پیدا کردن UID فولدر
    path_parts = row["cropped image file path"].split("/")
    if len(path_parts) < 2: continue
    uid_path = path_parts[-2]
    
    uid_folder = os.path.join(JPEG_ROOT, uid_path)
    if not os.path.isdir(uid_folder): continue

    # لیست کردن فایل‌های jpg
    files = [f for f in os.listdir(uid_folder) if f.lower().endswith((".jpg", ".jpeg"))]
    
    selected_img_path = None
    
    # بررسی تک تک فایل‌های داخل فولدر
    for f in files:
        full_path = os.path.join(uid_folder, f)
        
        if is_valid_patch(full_path):
            selected_img_path = full_path
            break 
    
    if selected_img_path:
        # تعیین لیبل
        pathology = row["pathology"].strip().upper()
        label = 1 if "MALIGNANT" in pathology else 0
        
        data_rows.append([selected_img_path, label])
        stats["saved"] += 1
    else:
        # اگر هیچ پچ سالمی پیدا نشد، دلیلش را حدس می‌زنیم (برای آمار)
        # (این بخش فقط برای گزارش است و تاثیری در خروجی ندارد)
        stats["skipped_mask_only"] += 1 

# ذخیره خروجی
out_df = pd.DataFrame(data_rows, columns=["image_path", "label"])
out_df.to_csv(OUTPUT_CSV, index=False)

print("\n" + "="*40)
print(f"✅ DATASET GENERATED: {OUTPUT_CSV}")
print("="*40)
print(f"Total Valid Patches Saved: {stats['saved']}")
print(f"Skipped Cases (No valid patch found): {stats['skipped_mask_only']}")
print("-" * 30)
print("NOTE: The skipped cases likely contained only Masks or Giant Full Mammograms.")
print("Your dataset is now CLEAN and ready for training.")