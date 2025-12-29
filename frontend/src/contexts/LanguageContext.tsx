import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

type Language = 'en' | 'fa';

interface LanguageContextType {
  language: Language;
  setLanguage: (lang: Language) => void;
  t: (key: string) => string;
  dir: 'ltr' | 'rtl';
  isRTL: boolean;
}

const translations: Record<Language, Record<string, string>> = {
  en: {
    // Navigation
    'nav.dashboard': 'Dashboard',
    'nav.about': 'About Us',
    'nav.title': 'MicroCalc',
    'nav.subtitle': 'Breast Cancer Detection',

    // Dashboard - Upload
    'upload.title': 'Upload Mammography Patch',
    'upload.dragDrop': 'Drag & drop your image here',
    'upload.or': 'or',
    'upload.browse': 'Browse Files',
    'upload.supported': 'Supported formats: PNG, JPG, DICOM',
    'upload.maxSize': 'Max file size: 10MB',

    // Dashboard - Settings
    'settings.title': 'Advanced Settings',
    'settings.preprocessing': 'Preprocessing',
    'settings.clahe': 'CLAHE Enhancement',
    'settings.claheDesc': 'Apply Contrast Limited Adaptive Histogram Equalization',
    'settings.model': 'Model Selection',
    'settings.modelDesc': 'Choose AI architecture for analysis',
    'settings.threshold': 'Confidence Threshold',
    'settings.thresholdDesc': 'Adjust sensitivity for classification',

    // Dashboard - Analysis
    'analyze.button': 'Analyze Image',
    'analyze.analyzing': 'Analyzing...',
    'analyze.noImage': 'Please upload an image first',

    // Dashboard - Results
    'results.title': 'Analysis Results',
    'results.original': 'Original Image',
    'results.gradcam': 'Grad-CAM Heatmap',
    'results.prediction': 'Prediction',
    'results.confidence': 'Confidence',
    'results.benign': 'Benign',
    'results.malignant': 'Malignant',
    'results.uncertain': 'Uncertain',
    'results.warning': 'Low Confidence Warning',
    'results.warningDesc': 'The model confidence is between 45-55%. This result should be reviewed by a specialist.',
    'results.export': 'Export Results',
    'results.exportPNG': 'Export as PNG',
    'results.noResults': 'Upload and analyze an image to see results',

    // Dashboard - Metadata
    'meta.filename': 'File Name',
    'meta.dimensions': 'Dimensions',
    'meta.filesize': 'File Size',
    'meta.model': 'Model Used',

    // Models
    'model.efficientnet': 'EfficientNet-B0',
    'model.resnet': 'ResNet18',
    'model.custom': 'Custom Path',

    // About Page
    'about.projectTitle': 'About MicroCalc',
    'about.projectDesc': 'MicroCalc is a cutting-edge research project focused on detecting breast cancer microcalcifications in mammography images using deep learning and explainable AI techniques.',
    'about.methodology': 'Our Methodology',
    'about.methodologyDesc': 'Our pipeline combines state-of-the-art image preprocessing with advanced neural network architectures to provide accurate and interpretable results.',
    'about.step1Title': 'CLAHE Preprocessing',
    'about.step1Desc': 'Contrast Limited Adaptive Histogram Equalization enhances local image contrast for better feature visibility.',
    'about.step2Title': 'CNN Classification',
    'about.step2Desc': 'Deep convolutional neural networks (EfficientNet-B0, ResNet18) classify patches as benign or malignant.',
    'about.step3Title': 'Grad-CAM Explainability',
    'about.step3Desc': 'Gradient-weighted Class Activation Mapping provides visual explanations of model decisions.',
    'about.team': 'Our Team',
    'about.teamDesc': 'Meet the dedicated researchers behind MicroCalc',
    'about.role.datascientist': 'Data Scientist',
    'about.role.fullstack': 'Full Stack Developer',
    'about.role.mleng': 'ML Engineer',
    'about.role.uiux': 'UI/UX Designer',
    'about.university': 'University Capstone Project',

    // Common
    'common.loading': 'Loading...',
    'common.error': 'An error occurred',
    'common.success': 'Success',
  },
  fa: {
    // Navigation
    'nav.dashboard': 'داشبورد',
    'nav.about': 'درباره ما',
    'nav.title': 'میکروکلک',
    'nav.subtitle': 'تشخیص سرطان پستان',

    // Dashboard - Upload
    'upload.title': 'بارگذاری تصویر ماموگرافی',
    'upload.dragDrop': 'تصویر خود را اینجا بکشید و رها کنید',
    'upload.or': 'یا',
    'upload.browse': 'انتخاب فایل',
    'upload.supported': 'فرمت‌های پشتیبانی: PNG، JPG، DICOM',
    'upload.maxSize': 'حداکثر حجم فایل: ۱۰ مگابایت',

    // Dashboard - Settings
    'settings.title': 'تنظیمات پیشرفته',
    'settings.preprocessing': 'پیش‌پردازش',
    'settings.clahe': 'بهبود CLAHE',
    'settings.claheDesc': 'اعمال یکسان‌سازی هیستوگرام تطبیقی با محدودیت کنتراست',
    'settings.model': 'انتخاب مدل',
    'settings.modelDesc': 'معماری هوش مصنوعی را برای تحلیل انتخاب کنید',
    'settings.threshold': 'آستانه اطمینان',
    'settings.thresholdDesc': 'تنظیم حساسیت برای طبقه‌بندی',

    // Dashboard - Analysis
    'analyze.button': 'تحلیل تصویر',
    'analyze.analyzing': 'در حال تحلیل...',
    'analyze.noImage': 'لطفاً ابتدا یک تصویر بارگذاری کنید',

    // Dashboard - Results
    'results.title': 'نتایج تحلیل',
    'results.original': 'تصویر اصلی',
    'results.gradcam': 'نقشه حرارتی Grad-CAM',
    'results.prediction': 'پیش‌بینی',
    'results.confidence': 'اطمینان',
    'results.benign': 'خوش‌خیم',
    'results.malignant': 'بدخیم',
    'results.uncertain': 'نامشخص',
    'results.warning': 'هشدار اطمینان پایین',
    'results.warningDesc': 'اطمینان مدل بین ۴۵ تا ۵۵ درصد است. این نتیجه باید توسط متخصص بررسی شود.',
    'results.export': 'خروجی نتایج',
    'results.exportPNG': 'خروجی PNG',
    'results.noResults': 'برای مشاهده نتایج، تصویر را بارگذاری و تحلیل کنید',

    // Dashboard - Metadata
    'meta.filename': 'نام فایل',
    'meta.dimensions': 'ابعاد',
    'meta.filesize': 'حجم فایل',
    'meta.model': 'مدل استفاده شده',

    // Models
    'model.efficientnet': 'EfficientNet-B0',
    'model.resnet': 'ResNet18',
    'model.custom': 'مسیر سفارشی',

    // About Page
    'about.projectTitle': 'درباره میکروکلک',
    'about.projectDesc': 'میکروکلک یک پروژه تحقیقاتی پیشرفته است که بر تشخیص میکروکلسیفیکاسیون‌های سرطان پستان در تصاویر ماموگرافی با استفاده از یادگیری عمیق و تکنیک‌های هوش مصنوعی قابل تفسیر تمرکز دارد.',
    'about.methodology': 'روش‌شناسی ما',
    'about.methodologyDesc': 'خط لوله ما پیش‌پردازش تصویر پیشرفته را با معماری‌های شبکه عصبی پیشرفته ترکیب می‌کند تا نتایج دقیق و قابل تفسیر ارائه دهد.',
    'about.step1Title': 'پیش‌پردازش CLAHE',
    'about.step1Desc': 'یکسان‌سازی هیستوگرام تطبیقی با محدودیت کنتراست، کنتراست محلی تصویر را برای دید بهتر ویژگی‌ها بهبود می‌بخشد.',
    'about.step2Title': 'طبقه‌بندی CNN',
    'about.step2Desc': 'شبکه‌های عصبی کانولوشنی عمیق (EfficientNet-B0، ResNet18) پچ‌ها را به عنوان خوش‌خیم یا بدخیم طبقه‌بندی می‌کنند.',
    'about.step3Title': 'تفسیرپذیری Grad-CAM',
    'about.step3Desc': 'نقشه‌برداری فعال‌سازی کلاس وزن‌دار گرادیان، توضیحات بصری از تصمیمات مدل ارائه می‌دهد.',
    'about.team': 'تیم ما',
    'about.teamDesc': 'با محققان متعهد پشت میکروکلک آشنا شوید',
    'about.role.datascientist': 'دانشمند داده',
    'about.role.fullstack': 'توسعه‌دهنده فول‌استک',
    'about.role.mleng': 'مهندس یادگیری ماشین',
    'about.role.uiux': 'طراح UI/UX',
    'about.university': 'پروژه پایان‌نامه دانشگاهی',

    // Common
    'common.loading': 'در حال بارگذاری...',
    'common.error': 'خطایی رخ داد',
    'common.success': 'موفقیت',
  },
};

const LanguageContext = createContext<LanguageContextType | undefined>(undefined);

export const LanguageProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [language, setLanguageState] = useState<Language>(() => {
    const saved = localStorage.getItem('microcalc-language');
    return (saved as Language) || 'en';
  });

  const setLanguage = (lang: Language) => {
    setLanguageState(lang);
    localStorage.setItem('microcalc-language', lang);
  };

  const t = (key: string): string => {
    return translations[language][key] || key;
  };

  const dir = language === 'fa' ? 'rtl' : 'ltr';
  const isRTL = language === 'fa';

  useEffect(() => {
    document.documentElement.setAttribute('dir', dir);
    document.documentElement.setAttribute('lang', language);
  }, [dir, language]);

  return (
    <LanguageContext.Provider value={{ language, setLanguage, t, dir, isRTL }}>
      {children}
    </LanguageContext.Provider>
  );
};

export const useLanguage = () => {
  const context = useContext(LanguageContext);
  if (!context) {
    throw new Error('useLanguage must be used within a LanguageProvider');
  }
  return context;
};