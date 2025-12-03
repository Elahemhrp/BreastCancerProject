import pandas as pd
import os
import glob
from tqdm import tqdm # برای نمایش نوار پیشرفت

# ==========================================
# تنظیمات مسیرها (بر اساس عکس ارسالی شما)
# ==========================================
# فرض بر این است که اسکریپت کنار پوشه data اجرا می‌شود
CSV_PATH = "./data/csv/calc_case_description_train_set.csv"
JPEG_ROOT_DIR = "./data/jpeg"

def deep_audit():
    print(">>> STARTING DEEP FORENSIC AUDIT <<<")
    print("="*60)

    # -------------------------------------------------------
    # گام ۱: سرشماری فیزیکی (آنچه واقعاً روی هارد دارید)
    # -------------------------------------------------------
    print(f"[Step 1] Scanning physical files in {JPEG_ROOT_DIR}...")
    
    if not os.path.exists(JPEG_ROOT_DIR):
        print(f"CRITICAL ERROR: Folder {JPEG_ROOT_DIR} does not exist!")
        return

    # دیکشنری برای نگهداری آدرس فایل‌های پیدا شده
    # Key: نام پوشه UID (مثلا 1.3.6.1...), Value: لیست فایل‌های داخلش
    physical_inventory = {}
    total_jpg_files = 0
    
    # پیمایش تمام پوشه‌ها
    uids = os.listdir(JPEG_ROOT_DIR)
    for uid_folder in tqdm(uids, desc="Scanning Folders"):
        full_folder_path = os.path.join(JPEG_ROOT_DIR, uid_folder)
        
        if os.path.isdir(full_folder_path):
            # پیدا کردن فایل‌های تصویر داخل پوشه
            images = [f for f in os.listdir(full_folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            
            if len(images) > 0:
                physical_inventory[uid_folder] = images
                total_jpg_files += len(images)

    print(f"\n>> PHYSICAL SCAN RESULTS:")
    print(f"   - Total Folders Scanned: {len(uids)}")
    print(f"   - Folders containing images: {len(physical_inventory)}")
    print(f"   - Total Image Files Found: {total_jpg_files}")
    print("="*60)

    # -------------------------------------------------------
    # گام ۲: خواندن نقشه (فایل CSV)
    # -------------------------------------------------------
    print(f"[Step 2] Reading the CSV Map ({CSV_PATH})...")
    try:
        df = pd.read_csv(CSV_PATH)
        print(f"   - Total Rows in CSV: {len(df)}")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # -------------------------------------------------------
    # گام ۳: تطابق (Cross-Referencing)
    # -------------------------------------------------------
    print("\n[Step 3] Cross-Referencing CSV vs. Disk...")
    
    matched_records = []     # رکوردهایی که عکسشان پیدا شد
    missing_records = []     # رکوردهایی که عکسشان نیست
    files_per_folder_stats = {} # آمار اینکه تو هر پوشه چند فایل بود

    # پیمایش روی CSV
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Verifying"):
        csv_path = row['cropped image file path'] # مسیر طولانی داخل CSV
        
        # استخراج UID از مسیر CSV (پیدا کردن بخشی که شبیه نام پوشه است)
        # مسیر معمولا این شکلیه: Calc-Training_.../UID_SERIES/UID_IMAGE/...
        path_parts = csv_path.split('/')
        
        found_uid = None
        for part in path_parts:
            # چک می‌کنیم آیا این تیکه از آدرس، به عنوان پوشه در هارد ما هست؟
            if part in physical_inventory:
                found_uid = part
                break
        
        if found_uid:
            # پیدا شد!
            file_count = len(physical_inventory[found_uid])
            files_per_folder_stats[file_count] = files_per_folder_stats.get(file_count, 0) + 1
            
            # ذخیره اطلاعات برای آمارگیری نهایی
            record_info = {
                'patient_id': row['patient_id'],
                'pathology': row['pathology'],
                'calc_type': row['calc type'],
                'calc_distribution': row['calc distribution'],
                'found_files': file_count
            }
            matched_records.append(record_info)
        else:
            # پیدا نشد
            missing_records.append(row['patient_id'])

    # تبدیل لیست مچ شده‌ها به DataFrame برای آنالیز راحت‌تر
    df_matched = pd.DataFrame(matched_records)

    # -------------------------------------------------------
    # گام ۴: گزارش نهایی (از شیر مرغ تا جان آدمیزاد)
    # -------------------------------------------------------
    print("\n" + "="*60)
    print(">>> FINAL FORENSIC REPORT <<<")
    print("="*60)
    
    print(f"1. DATA INTEGRITY (سلامت داده‌ها):")
    print(f"   - Expected Images (from CSV): {len(df)}")
    print(f"   - Actual Images Found (Linked): {len(df_matched)}")
    print(f"   - Missing/Broken Links: {len(missing_records)}")
    
    if len(df_matched) == 0:
        print("\n!!! CRITICAL: NO MATCHES FOUND. CHECK PATHS !!!")
        return

    accuracy = (len(df_matched) / len(df)) * 100
    print(f"   - Dataset Integrity Score: {accuracy:.2f}%")
    
    print("-" * 40)
    print(f"2. FOLDER CONTENTS (داخل پوشه‌ها چه خبر است؟):")
    for count, frequency in files_per_folder_stats.items():
        print(f"   - Folders containing exactly {count} image(s): {frequency} folders")
    
    if 1 in files_per_folder_stats and len(files_per_folder_stats) == 1:
        print("   >> RESULT: Perfect! Each folder contains exactly 1 image.")
    else:
        print("   >> NOTE: Some folders have multiple images (or are empty).")

    print("-" * 40)
    print(f"3. CLINICAL STATISTICS (آمار پزشکی روی داده‌های موجود):")
    
    # تمیز کردن لیبل‌ها
    def clean_label(l): return "Malignant" if "MALIGNANT" in l else "Benign"
    df_matched['simple_label'] = df_matched['pathology'].apply(clean_label)
    
    counts = df_matched['simple_label'].value_counts()
    print("\n   [A] Class Balance (Benign vs Malignant):")
    print(counts.to_string())
    print(f"   - Malignant Ratio: {(counts.get('Malignant',0)/len(df_matched)*100):.1f}%")

    print("\n   [B] Top 5 Calcification Types (شکل ظاهری):")
    print(df_matched['calc_type'].value_counts().head(5).to_string())

    print("\n   [C] Patients Count:")
    print(f"   - Total Unique Patients in valid data: {df_matched['patient_id'].nunique()}")

    print("="*60)
    print("Analysis Complete.")

if __name__ == "__main__":
    deep_audit()