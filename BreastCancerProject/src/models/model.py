import torch
import torch.nn as nn
from torchvision import models

def get_model(device="cpu"):
    # 1. دانلود مدل ResNet18 (نسخه سبک و سریع)
    # weights='DEFAULT' یعنی وزن‌های آموزش دیده روی ImageNet را دانلود کن
    model = models.resnet18(weights='DEFAULT')
    
    # 2. جراحی لایه اول (برای ورودی سیاه و سفید)
    # لایه اصلی: nn.Conv2d(3, 64, ...) -> ورودی 3 کانال
    # تغییر به: nn.Conv2d(1, 64, ...) -> ورودی 1 کانال
    # وزن‌های کانال‌های رنگی را میانگین می‌گیریم تا اطلاعات از دست نرود
    original_weights = model.conv1.weight.data.clone()
    new_weights = original_weights.mean(dim=1, keepdim=True)
    
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.conv1.weight.data = new_weights
    
    # 3. جراحی لایه آخر (برای خروجی 2 کلاسه: خوش‌خیم/بدخیم)
    num_ftrs = model.fc.in_features
    # خروجی 1 عدد (Logit) می‌دهیم. اگر مثبت بود -> بدخیم، منفی -> خوش‌خیم
    model.fc = nn.Linear(num_ftrs, 1)
    
    return model.to(device)