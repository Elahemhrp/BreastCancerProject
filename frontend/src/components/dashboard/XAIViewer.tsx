import React from 'react';
import { useLanguage } from '@/contexts/LanguageContext';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { ImageOff } from 'lucide-react';
import { cn } from '@/lib/utils';

interface XAIViewerProps {
  originalImage: string | null;
  gradcamImage: string | null;
  isLoading?: boolean;
}

const XAIViewer: React.FC<XAIViewerProps> = ({ originalImage, gradcamImage, isLoading }) => {
  const { t, isRTL } = useLanguage();

  const ImagePlaceholder = ({ label }: { label: string }) => (
    <div className="flex flex-col items-center justify-center h-full min-h-[280px] bg-secondary/30 rounded-lg border-2 border-dashed border-border">
      <ImageOff className="h-12 w-12 text-muted-foreground/50 mb-3" />
      <p className="text-sm text-muted-foreground">{label}</p>
    </div>
  );

  const LoadingSkeleton = () => (
    <div className="space-y-3">
      <Skeleton className="w-full h-[280px] rounded-lg" />
    </div>
  );

  return (
    <div className={cn(
      "grid grid-cols-1 md:grid-cols-2 gap-4 h-full",
      isRTL && "md:grid-flow-col-dense"
    )}>
      {/* Original Image */}
      <Card className="overflow-hidden shadow-soft">
        <CardHeader className="py-3 px-4 bg-secondary/30 border-b border-border">
          <CardTitle className={cn("text-sm font-medium", isRTL && "text-right")}>
            {t('results.original')}
          </CardTitle>
        </CardHeader>
        <CardContent className="p-4">
          {isLoading ? (
            <LoadingSkeleton />
          ) : originalImage ? (
            <div className="relative rounded-lg overflow-hidden bg-secondary/20 animate-fade-in">
              <img
                src={originalImage}
                alt="Original mammography"
                className="w-full h-auto object-contain max-h-[320px]"
              />
            </div>
          ) : (
            <ImagePlaceholder label={t('results.noResults')} />
          )}
        </CardContent>
      </Card>

      {/* Grad-CAM Heatmap */}
      <Card className="overflow-hidden shadow-soft">
        <CardHeader className="py-3 px-4 bg-accent/10 border-b border-accent/20">
          <CardTitle className={cn("text-sm font-medium text-accent-foreground", isRTL && "text-right")}>
            {t('results.gradcam')}
          </CardTitle>
        </CardHeader>
        <CardContent className="p-4">
          {isLoading ? (
            <LoadingSkeleton />
          ) : gradcamImage ? (
            <div className="relative rounded-lg overflow-hidden bg-secondary/20 animate-fade-in">
              <img
                src={gradcamImage}
                alt="Grad-CAM heatmap"
                className="w-full h-auto object-contain max-h-[320px]"
              />
              <div className="absolute bottom-2 right-2 bg-background/80 backdrop-blur-sm px-2 py-1 rounded text-xs text-muted-foreground">
                Grad-CAM
              </div>
            </div>
          ) : (
            <ImagePlaceholder label={t('results.noResults')} />
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default XAIViewer;