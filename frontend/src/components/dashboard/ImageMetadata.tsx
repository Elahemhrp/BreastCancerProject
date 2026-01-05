import React from 'react';
import { useLanguage } from '@/contexts/LanguageContext';
import { FileImage, Ruler, HardDrive, Cpu } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ImageMetadataProps {
  filename: string | null;
  dimensions: { width: number; height: number } | null;
  filesize: number | null;
  modelUsed: string | null;
}

const ImageMetadata: React.FC<ImageMetadataProps> = ({
  filename,
  dimensions,
  filesize,
  modelUsed,
}) => {
  const { t, isRTL } = useLanguage();

  if (!filename) return null;

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const items = [
    { icon: FileImage, label: t('meta.filename'), value: filename },
    { 
      icon: Ruler, 
      label: t('meta.dimensions'), 
      value: dimensions ? `${dimensions.width} × ${dimensions.height} px` : null 
    },
    { 
      icon: HardDrive, 
      label: t('meta.filesize'), 
      value: filesize ? formatFileSize(filesize) : null 
    },
    { icon: Cpu, label: t('meta.model'), value: modelUsed },
  ].filter(item => item.value);

  return (
    <div className={cn(
      "flex flex-wrap gap-4 p-3 rounded-lg bg-secondary/30 border border-border text-xs",
      isRTL && "flex-row-reverse"
    )}>
      {items.map((item, index) => {
        const Icon = item.icon;
        return (
          <div 
            key={index} 
            className={cn(
              "flex items-center gap-1.5 text-muted-foreground",
              isRTL && "flex-row-reverse"
            )}
          >
            <Icon className="h-3.5 w-3.5" />
            <span>{item.label}:</span>
            <span className="text-foreground font-medium">{item.value}</span>
          </div>
        );
      })}
    </div>
  );
};

export default ImageMetadata;