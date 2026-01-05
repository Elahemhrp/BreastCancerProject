import React, { useState, useCallback } from 'react';
import { useLanguage } from '@/contexts/LanguageContext';
import { Upload, Image as ImageIcon, X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

interface ImageUploadProps {
  onImageSelect: (file: File, preview: string) => void;
  selectedImage: string | null;
  onClear: () => void;
}

const ImageUpload: React.FC<ImageUploadProps> = ({ onImageSelect, selectedImage, onClear }) => {
  const { t, isRTL } = useLanguage();
  const [isDragging, setIsDragging] = useState(false);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDragIn = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.dataTransfer.items && e.dataTransfer.items.length > 0) {
      setIsDragging(true);
    }
  }, []);

  const handleDragOut = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      processFile(file);
    }
  }, []);

  const processFile = (file: File) => {
    if (!file.type.startsWith('image/')) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      const preview = e.target?.result as string;
      onImageSelect(file, preview);
    };
    reader.readAsDataURL(file);
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      processFile(e.target.files[0]);
    }
  };

  if (selectedImage) {
    return (
      <div className="relative rounded-xl overflow-hidden border border-border bg-card animate-fade-in">
        <img
          src={selectedImage}
          alt="Selected mammography patch"
          className="w-full h-48 object-contain bg-secondary/30"
        />
        <Button
          variant="destructive"
          size="icon"
          className="absolute top-2 right-2 h-8 w-8 rounded-full shadow-lg"
          onClick={onClear}
        >
          <X className="h-4 w-4" />
        </Button>
      </div>
    );
  }

  return (
    <div
      className={cn(
        "upload-zone cursor-pointer group",
        isDragging && "dragging"
      )}
      onDragEnter={handleDragIn}
      onDragLeave={handleDragOut}
      onDragOver={handleDrag}
      onDrop={handleDrop}
      onClick={() => document.getElementById('file-input')?.click()}
    >
      <input
        id="file-input"
        type="file"
        accept="image/*"
        onChange={handleFileSelect}
        className="hidden"
      />
      
      <div className={cn("flex flex-col items-center gap-4", isRTL && "text-content")}>
        <div className="relative">
          <div className="absolute inset-0 bg-accent/20 rounded-full blur-xl group-hover:blur-2xl transition-all" />
          <div className="relative bg-secondary rounded-full p-4 group-hover:bg-accent/20 transition-colors">
            <Upload className="h-8 w-8 text-muted-foreground group-hover:text-accent-foreground transition-colors" />
          </div>
        </div>

        <div className="text-center space-y-1">
          <p className="font-medium text-foreground">
            {t('upload.dragDrop')}
          </p>
          <p className="text-sm text-muted-foreground">
            {t('upload.or')}
          </p>
          <Button variant="outline" size="sm" className="mt-2" type="button">
            <ImageIcon className="h-4 w-4 mr-2" />
            {t('upload.browse')}
          </Button>
        </div>

        <div className="text-xs text-muted-foreground text-center space-y-0.5">
          <p>{t('upload.supported')}</p>
          <p>{t('upload.maxSize')}</p>
        </div>
      </div>
    </div>
  );
};

export default ImageUpload;