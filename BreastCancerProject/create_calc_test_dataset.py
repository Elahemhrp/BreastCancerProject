import os
import pandas as pd
import cv2
import numpy as np

# ================= تنظیمات مسیر =================
BASE_DIR = os.getcwd()
# تغییر مهم: فایل ورودی تست است
CSV_PATH = os.path.join(BASE_DIR, "data/csv/calc_case_description_test_set.csv")
JPEG_ROOT = os.path.join(BASE_DIR, "data/jpeg")
# تغییر مهم: نام خروجی تست است
OUTPUT_CSV = "calc_test_paths.csv"

# ================= توابع فیلتر (دقیقا مثل Train) =================
def is_valid_patch(img_path):
    """
    بررسی می‌کند که عکس:
    1. ماسک نباشد (تعداد رنگ > 30).
    2. عکس کامل نباشد (ابعاد < 2000).
    """
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: return False
        
        h, w = img.shape
        unique_vals = len(np.unique(img))
        
        # فیلتر ماسک
        if unique_vals < 30:
            return False
            
        # فیلتر عکس خیلی بزرگ (Full Mammogram)
        if h > 2000 or w > 2000:
            return False
            
        return True
    except Exception:
        return False

# ================= بدنه اصلی =================
print("🚀 Starting FINAL TEST Dataset Creation...")
if not os.path.exists(CSV_PATH):
    print(f"❌ Error: CSV file not found at {CSV_PATH}")
    exit()

df = pd.read_csv(CSV_PATH)

data_rows = []
stats = {
    "saved": 0,
    "skipped": 0
}

for idx, row in df.iterrows():
    # استخراج UID از ستون cropped image file path
    path_parts = row["cropped image file path"].split("/")
    if len(path_parts) < 2: continue
    uid_path = path_parts[-2]
    
    uid_folder = os.path.join(JPEG_ROOT, uid_path)
    if not os.path.isdir(uid_folder): continue

    files = [f for f in os.listdir(uid_folder) if f.lower().endswith((".jpg", ".jpeg"))]
    
    selected_img_path = None
    
    # جستجوی پچ سالم بین فایل‌ها
    for f in files:
        full_path = os.path.join(uid_folder, f)
        if is_valid_patch(full_path):
            selected_img_path = full_path
            break 
    
    if selected_img_path:
        pathology = row["pathology"].strip().upper()
        label = 1 if "MALIGNANT" in pathology else 0
        
        data_rows.append([selected_img_path, label])
        stats["saved"] += 1
    else:
        stats["skipped"] += 1

# ذخیره خروجی
out_df = pd.DataFrame(data_rows, columns=["image_path", "label"])
out_df.to_csv(OUTPUT_CSV, index=False)

print("\n" + "="*40)
print(f"✅ TEST DATASET GENERATED: {OUTPUT_CSV}")
print("="*40)
print(f"Total Valid Patches: {stats['saved']}")
print(f"Skipped Cases:       {stats['skipped']}")
print("-" * 30)