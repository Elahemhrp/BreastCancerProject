import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# ================= تنظیمات مسیر هوشمند (Smart Path Setup) =================
# 1. پیدا کردن مسیر فایلی که الان دارد اجرا می‌شود (train_model.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. پیدا کردن ریشه پروژه (دو مرحله عقب‌گرد: src/models -> src -> root)
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, '../../'))

# 3. اضافه کردن ریشه به مسیرهای پایتون (برای ایمپورت کردن src)
sys.path.append(PROJECT_ROOT)

# 4. ساخت مسیرهای قطعی برای فایل‌های CSV
TRAIN_CSV_PATH = os.path.join(PROJECT_ROOT, 'data', 'calc_train_paths.csv')
TEST_CSV_PATH = os.path.join(PROJECT_ROOT, 'data', 'calc_test_paths.csv')

# چک کردن اینکه فایل‌ها واقعا وجود دارند (برای دیباگ)
if not os.path.exists(TRAIN_CSV_PATH):
    raise FileNotFoundError(f"❌ Error: Could not find train CSV at: {TRAIN_CSV_PATH}")

# ================= ایمپورت ماژول‌های پروژه =================
from src.preprocess.dataset import build_dataloaders
from src.models.model import get_model

# ================= تنظیمات مدل =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 16 

# ... (بقیه توابع train_one_epoch و evaluate مثل قبل هستند) ...

def train_one_epoch(model, loader, criterion, optimizer):
    model.train() # حالت آموزش (DropOut فعال، BatchNormal فعال)
    running_loss = 0.0
    correct = 0
    total = 0
    
    loop = tqdm(loader, desc="Training")
    
    for images, labels in loop:
        images, labels = images.to(DEVICE), labels.to(DEVICE).float()
        
        # 1. Forward Pass
        outputs = model(images).squeeze(1) # خروجی مدل
        loss = criterion(outputs, labels)  # محاسبه خطا
        
        # 2. Backward Pass
        optimizer.zero_grad() # صفر کردن گرادیان‌های قبلی
        loss.backward()       # محاسبه گرادیان جدید
        optimizer.step()      # آپدیت وزن‌ها
        
        # آمارگیری
        running_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float() # تبدیل احتمالات به 0 و 1
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        loop.set_postfix(loss=loss.item())
        
    return running_loss / len(loader), correct / total

def evaluate(model, loader, criterion):
    model.eval() # حالت ارزیابی (بدون آپدیت وزن)
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad(): # گرادیان نگیر (رم صرفه‌جویی میشه)
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE).float()
            
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    return running_loss / len(loader), correct / total

def main():
    print(f"🚀 Training on {DEVICE}...")
    print(f"📂 Reading data from: {PROJECT_ROOT}/data")
    
    # استفاده از مسیرهای قطعی که بالا تعریف کردیم
    train_loader, test_loader = build_dataloaders(
        train_csv=TRAIN_CSV_PATH,  # <--- تغییر مهم
        test_csv=TEST_CSV_PATH,    # <--- تغییر مهم
        batch_size=BATCH_SIZE
    )
    
    # 2. ساخت مدل
    model = get_model(DEVICE)
    
    # 3. تعریف Loss و Optimizer
    # BCEWithLogitsLoss برای دسته‌بندی باینری عالیه (پایدارتر از BCELoss)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_acc = 0.0
    
    # 4. حلقه اصلی آموزش
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, test_loader, criterion)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        
        # ذخیره بهترین مدل
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "models/best_model.pth")
            print("✅ New Best Model Saved!")

if __name__ == "__main__":
    main()