import pandas as pd
import os

# تنظیمات
CSV_PATH = "./data/csv/calc_case_description_train_set.csv"
JPEG_ROOT = "./data/jpeg"

# لیست پوشه‌هایی که شما (هادی) به صورت دستی چک کردید و دیدید 1 فایل دارند
USER_FOUND_FOLDERS = [
    "1.3.6.1.4.1.9590.100.1.2.2668955613180149841960621060291212705",
    "1.3.6.1.4.1.9590.100.1.2.4164835311517392213293157733287891901",
    "1.3.6.1.4.1.9590.100.1.2.4071275011184216523502721732765963066",
    "1.3.6.1.4.1.9590.100.1.2.5650664511554743340616290601354671803",
    "1.3.6.1.4.1.9590.100.1.2.6860571612629209739046762663550661960"
]

def resolve_conflict():
    print(">>> CONFLICT RESOLUTION SCRIPT <<<")
    print("-" * 50)
    
    # 1. لود کردن CSV
    try:
        df = pd.read_csv(CSV_PATH)
        print(f"CSV Loaded. Total rows (Target Data): {len(df)}")
    except:
        print("Error loading CSV.")
        return

    # استخراج تمام UIDهای موجود در CSV
    # در CSV آدرس‌ها طولانی هستند، ما باید تک تک اجزای آدرس را چک کنیم
    csv_uids = set()
    for path in df['cropped image file path']:
        parts = path.split('/')
        for part in parts:
            csv_uids.add(part)
            
    print(f"Total Unique Folder IDs in CSV: {len(csv_uids)}")
    print("-" * 50)

    # 2. بررسی پوشه‌های پیدا شده توسط شما (هادی)
    print("Checking the specific folders YOU found:")
    match_count = 0
    for folder in USER_FOUND_FOLDERS:
        status = "UNKNOWN"
        if folder in csv_uids:
            status = "✅ IN TARGET LIST (Training Data)"
            match_count += 1
        else:
            status = "❌ IRRELEVANT (Test Data or Full Mammogram)"
        
        # چک کردن محتویات واقعی روی دیسک
        real_path = os.path.join(JPEG_ROOT, folder)
        file_count = "N/A"
        if os.path.exists(real_path):
            files = os.listdir(real_path)
            file_count = len(files)
            
        print(f"ID: ...{folder[-10:]} | Status: {status} | Files inside: {file_count}")

    print("-" * 50)

    # 3. بررسی معکوس: انتخاب چند نمونه واقعی از CSV و چک کردن دیسک
    print("Checking random samples specifically FROM THE CSV:")
    
    # برداشتن اولین ردیف CSV برای تست
    sample_row = df.iloc[0] 
    csv_full_path = sample_row['cropped image file path']
    print(f"\nTarget CSV Path: {csv_full_path}")
    
    # تلاش برای پیدا کردن پوشه این ردیف روی دیسک
    parts = csv_full_path.split('/')
    found_on_disk = False
    for part in parts:
        potential_path = os.path.join(JPEG_ROOT, part)
        if os.path.exists(potential_path) and os.path.isdir(potential_path):
            files = os.listdir(potential_path)
            print(f">> Folder Found on Disk: {part}")
            print(f">> Files inside this target folder: {files}")
            found_on_disk = True
            break
            
    if not found_on_disk:
        print(">> Could not find the specific folder for this CSV row on disk.")

if __name__ == "__main__":
    resolve_conflict()