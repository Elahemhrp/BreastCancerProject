# import os
# import pandas as pd
# import cv2
# import numpy as np
# from collections import Counter
# from tqdm import tqdm # برای نمایش نوار پیشرفت

# # ================= تنظیمات مسیر =================
# BASE_DIR = os.getcwd()
# CSV_PATH = os.path.join(BASE_DIR, "data/csv/calc_case_description_train_set.csv")
# JPEG_ROOT = os.path.join(BASE_DIR, "data/jpeg")

# def analyze_image(img_path):
#     """
#     آنالیز یک عکس برای تشخیص نوع آن
#     """
#     try:
#         # فقط هدر فایل را نمی‌خوانیم، خود پیکسل‌ها مهم هستند برای شمارش رنگ
#         # برای سرعت، به صورت سیاه و سفید می‌خوانیم
#         img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#         if img is None:
#             return None
        
#         h, w = img.shape
#         unique_vals = len(np.unique(img))
        
#         # منطق تشخیص
#         # 1. ماسک: رنگ‌های خیلی کم (معمولا زیر ۲۰ تا)
#         # 2. بافت (Patch): رنگ‌های زیاد (نویز و بافت بافت سینه)
#         img_type = "UNKNOWN"
#         if unique_vals < 50:
#             img_type = "MASK"
#         else:
#             img_type = "TISSUE (PATCH)"
            
#         return {
#             "status": "OK",
#             "height": h,
#             "width": w,
#             "unique_colors": unique_vals,
#             "type": img_type
#         }
#     except Exception as e:
#         return {"status": "ERROR", "error": str(e)}

# # ================= بدنه اصلی =================
# print(f"Reading CSV: {CSV_PATH}")
# df = pd.read_csv(CSV_PATH)

# print(f"Scanning {len(df)} folders... Please wait.")

# stats = {
#     "total_folders_checked": 0,
#     "missing_folders": 0,
#     "folders_with_0_imgs": 0,
#     "folders_with_1_imgs": 0,
#     "folders_with_2_imgs": 0,
#     "folders_with_more_imgs": 0,
#     "valid_patches_found": 0, # تعداد عکس‌های بافت واقعی پیدا شده
#     "only_mask_found": 0,     # فولدرهایی که فقط ماسک دارند (خطرناک)
# }

# # لیست برای ذخیره نمونه‌های عجیب جهت بررسی دستی
# anomalies = []

# for idx, row in tqdm(df.iterrows(), total=len(df)):
#     # استخراج UID
#     path_parts = row["cropped image file path"].split("/")
#     if len(path_parts) < 2: continue
#     uid_path = path_parts[-2]
    
#     folder_path = os.path.join(JPEG_ROOT, uid_path)
#     stats["total_folders_checked"] += 1
    
#     if not os.path.exists(folder_path):
#         stats["missing_folders"] += 1
#         continue
        
#     # لیست فایل‌ها
#     files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg"))]
#     count = len(files)
    
#     # آپدیت آمار تعداد فایل
#     if count == 0: stats["folders_with_0_imgs"] += 1
#     elif count == 1: stats["folders_with_1_imgs"] += 1
#     elif count == 2: stats["folders_with_2_imgs"] += 1
#     else: stats["folders_with_more_imgs"] += 1
    
#     # آنالیز دقیق محتویات
#     folder_contents = []
#     has_tissue = False
    
#     for f in files:
#         full_path = os.path.join(folder_path, f)
#         info = analyze_image(full_path)
        
#         if info and info["status"] == "OK":
#             folder_contents.append(f"{info['type']} ({info['width']}x{info['height']}, Colors:{info['unique_colors']})")
#             if info["type"] == "TISSUE (PATCH)":
#                 has_tissue = True
    
#     # بررسی وضعیت سلامتی فولدر
#     if has_tissue:
#         stats["valid_patches_found"] += 1
#     elif count > 0:
#         # فایل هست ولی بافت نیست (همه‌ش ماسکه!)
#         stats["only_mask_found"] += 1
#         if len(anomalies) < 10: # فقط ۱۰ تا نمونه نگه دار
#             anomalies.append({
#                 "patient_id": row["patient_id"],
#                 "folder": uid_path,
#                 "files_found": count,
#                 "details": folder_contents
#             })

# # ================= گزارش نهایی =================
# print("\n" + "="*40)
# print("📊 FINAL DATASET AUDIT REPORT")
# print("="*40)
# print(f"Total entries in CSV:      {len(df)}")
# print(f"Folders Checked:           {stats['total_folders_checked']}")
# print(f"Missing Folders (Disk):    {stats['missing_folders']}")
# print("-" * 30)
# print("📂 Folder Content Stats:")
# print(f"  - Empty Folders (0 imgs): {stats['folders_with_0_imgs']}")
# print(f"  - 1 Image Folders:        {stats['folders_with_1_imgs']}")
# print(f"  - 2 Image Folders:        {stats['folders_with_2_imgs']}")
# print(f"  - >2 Image Folders:       {stats['folders_with_more_imgs']}")
# print("-" * 30)
# print("✅ Data Health:")
# print(f"  - Usable Patches Found:   {stats['valid_patches_found']} (Keep these)")
# print(f"  - USELESS (Only Masks):   {stats['only_mask_found']} (Must Drop)")
# print("="*40)

# if anomalies:
#     print("\n⚠️ ANOMALY SAMPLES (Folders with files but NO TISSUE):")
#     for a in anomalies:
#         print(f"Patient {a['patient_id']}: Found {a['files_found']} files -> {a['details']}")

import os
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm

# ================= تنظیمات مسیر =================
BASE_DIR = os.getcwd()
CSV_PATH = os.path.join(BASE_DIR, "data/csv/calc_case_description_train_set.csv")
JPEG_ROOT = os.path.join(BASE_DIR, "data/jpeg")

# ================= منطق طبقه‌بندی =================
def identify_image_type(img_path):
    try:
        # خواندن عکس به صورت سیاه و سفید
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: return "CORRUPT", 0, 0
        
        h, w = img.shape
        pixel_count = h * w
        unique_vals = len(np.unique(img))
        
        # 1. منطق تشخیص ماسک (ROI)
        # ماسک‌ها باینری یا نزدیک به باینری هستند
        if unique_vals < 50:
            return "ROI_MASK", h, w
        
        # 2. منطق تشخیص عکس کامل (Full Mammogram)
        # عکس کامل معمولا رزولوشن خیلی بالایی دارد (مثلا بالای 3 مگاپیکسل)
        # آستانه فرضی: اگر عرض یا ارتفاع خیلی زیاد باشد
        if h > 2500 or w > 2000:
            return "FULL_MAMMOGRAM", h, w
            
        # 3. اگر ماسک نیست و خیلی بزرگ هم نیست، پس پچ است
        return "CROPPED_PATCH", h, w
        
    except Exception as e:
        return "ERROR", 0, 0

# ================= بدنه اصلی =================
print("🚀 Starting Strict Classification...")
df = pd.read_csv(CSV_PATH)

stats = {
    "ROI_MASK": 0,
    "FULL_MAMMOGRAM": 0,
    "CROPPED_PATCH": 0,
    "CORRUPT": 0,
    "folders_checked": 0
}

# لیستی برای ذخیره نمونه‌های Full Mammogram (اگر پیدا شد)
full_mammogram_samples = []

print(f"Analyzing {len(df)} folders...")

for idx, row in tqdm(df.iterrows(), total=len(df)):
    path_parts = row["cropped image file path"].split("/")
    if len(path_parts) < 2: continue
    uid_path = path_parts[-2]
    
    folder_path = os.path.join(JPEG_ROOT, uid_path)
    if not os.path.exists(folder_path): continue
    
    stats["folders_checked"] += 1
    
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")]
    
    folder_composition = []
    
    for f in files:
        full_path = os.path.join(folder_path, f)
        img_type, h, w = identify_image_type(full_path)
        
        stats[img_type] += 1
        folder_composition.append(img_type)
        
        if img_type == "FULL_MAMMOGRAM":
            if len(full_mammogram_samples) < 5:
                full_mammogram_samples.append(f"{uid_path}/{f} ({w}x{h})")

# ================= گزارش نهایی =================
print("\n" + "="*40)
print("📊 CLASSIFICATION RESULTS")
print("="*40)
print(f"Total Folders Checked: {stats['folders_checked']}")
print(f"Total Images Scanned:  {stats['ROI_MASK'] + stats['FULL_MAMMOGRAM'] + stats['CROPPED_PATCH']}")
print("-" * 30)
print(f"Types Found:")
print(f"  ⚫ ROI MASK:          {stats['ROI_MASK']}")
print(f"  🖼️ CROPPED PATCH:     {stats['CROPPED_PATCH']}")
print(f"  🏢 FULL MAMMOGRAM:    {stats['FULL_MAMMOGRAM']}")
print(f"  ❌ CORRUPT/ERROR:     {stats['CORRUPT']}")
print("="*40)

if stats['FULL_MAMMOGRAM'] > 0:
    print("\n⚠️ WARNING: FULL MAMMOGRAMS DETECTED!")
    print("Samples:", full_mammogram_samples)
else:
    print("\n✅ CONFIRMED: No Full Mammograms found in these folders.")
    print("   The logic holds: Folders contain only [Mask, Patch].")

# (ptorch_env) hadi@Asus-Tuf:~/Coding/ML/AI Project-Breast Cancer- MicroCalcification/data team/1st week$ /home/hadi/Coding/ML/ptorch_env/bin/python "/home/hadi/Coding/ML/AI Project-Breast Cancer- MicroCalcification/data team/1st week/audit_dataset.py"
# Reading CSV: /home/hadi/Coding/ML/AI Project-Breast Cancer- MicroCalcification/data team/1st week/data/csv/calc_case_description_train_set.csv
# Scanning 1546 folders... Please wait.
# 100%|███████████████████████████████████████████████████████████| 1546/1546 [03:22<00:00,  7.65it/s]

# ========================================
# 📊 FINAL DATASET AUDIT REPORT
# ========================================
# Total entries in CSV:      1546
# Folders Checked:           1546
# Missing Folders (Disk):    1
# ------------------------------
# 📂 Folder Content Stats:
#   - Empty Folders (0 imgs): 0
#   - 1 Image Folders:        1
#   - 2 Image Folders:        1544
#   - >2 Image Folders:       0
# ------------------------------
# ✅ Data Health:
#   - Usable Patches Found:   1544 (Keep these)
#   - USELESS (Only Masks):   1 (Must Drop)
# ========================================

# ⚠️ ANOMALY SAMPLES (Folders with files but NO TISSUE):
# Patient P_00474: Found 1 files -> ['MASK (3301x5326, Colors:15)']
# (ptorch_env) hadi@Asus-Tuf:~/Coding/ML/AI Project-Breast Cancer- MicroCalcification/data team/1st week$ /home/hadi/Coding/ML/ptorch_env/bin/python "/home/hadi/Coding/ML/AI Project-Breast Cancer- MicroCalcification/data team/1st week/audit_dataset.py"
# 🚀 Starting Strict Classification...
# Analyzing 1546 folders...
# 100%|███████████████████████████████████████████████████████████| 1546/1546 [03:21<00:00,  7.66it/s]

# ========================================
# 📊 CLASSIFICATION RESULTS
# ========================================
# Total Folders Checked: 1545
# Total Images Scanned:  3089
# ------------------------------
# Types Found:
#   ⚫ ROI MASK:          1545
#   🖼️ CROPPED PATCH:     1515
#   🏢 FULL MAMMOGRAM:    29
#   ❌ CORRUPT/ERROR:     0
# ========================================

# ⚠️ WARNING: FULL MAMMOGRAMS DETECTED!
# Samples: ['1.3.6.1.4.1.9590.100.1.2.242813816211590557526939203903179610078/1-075.jpg (2797x3033)', '1.3.6.1.4.1.9590.100.1.2.108998325811479398607974727033630566895/1-076.jpg (2781x3817)', '1.3.6.1.4.1.9590.100.1.2.129769924413933273629894066072749004848/1-077.jpg (853x2521)', '1.3.6.1.4.1.9590.100.1.2.3866969912785618842192247560363093126/1-164.jpg (2189x545)', '1.3.6.1.4.1.9590.100.1.2.328842164210353914520706810182126250091/1-169.jpg (2193x1273)']