import React, { useState } from 'react';
import { useLanguage } from '@/contexts/LanguageContext';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
import { Play, Loader2 } from 'lucide-react';
import ImageUpload from '@/components/dashboard/ImageUpload';
import AdvancedSettings, { AnalysisSettings } from '@/components/dashboard/AdvancedSettings';
import XAIViewer from '@/components/dashboard/XAIViewer';
import PredictionResult, { PredictionData } from '@/components/dashboard/PredictionResult';
import ExportButton from '@/components/dashboard/ExportButton';
import ImageMetadata from '@/components/dashboard/ImageMetadata';
import { cn } from '@/lib/utils';
import { useToast } from '@/hooks/use-toast';

const Dashboard: React.FC = () => {
  const { t, isRTL } = useLanguage();
  const { toast } = useToast();
  
  // --- State Definitions ---
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [originalImage, setOriginalImage] = useState<string | null>(null);
  const [gradcamImage, setGradcamImage] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictionData | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [settings, setSettings] = useState<AnalysisSettings>({
    claheEnabled: true,
    model: 'resnet', // Default to ResNet as per backend
    threshold: 50,
  });
  const [imageDimensions, setImageDimensions] = useState<{ width: number; height: number } | null>(null);

  // --- Handlers ---

  const handleImageSelect = (file: File, preview: string) => {
    setSelectedFile(file);
    setOriginalImage(preview);
    setGradcamImage(null);
    setPrediction(null);
    
    // Get image dimensions
    const img = new Image();
    img.onload = () => {
      setImageDimensions({ width: img.width, height: img.height });
    };
    img.src = preview;
  };

  const handleClear = () => {
    setSelectedFile(null);
    setOriginalImage(null);
    setGradcamImage(null);
    setPrediction(null);
    setImageDimensions(null);
  };

  const getModelName = (model: string): string => {
    switch (model) {
      case 'efficientnet': return t('model.efficientnet');
      case 'resnet': return t('model.resnet');
      case 'custom': return t('model.custom');
      default: return model;
    }
  };

  // --- MAIN INTEGRATION FUNCTION ---
  const handleAnalyze = async () => {
    if (!originalImage || !selectedFile) {
      toast({
        title: t('analyze.noImage'),
        variant: 'destructive',
      });
      return;
    }

    setIsAnalyzing(true);
    setGradcamImage(null); 
    setPrediction(null);

    // 1. Prepare Form Data
    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      // 2. Send Request to Python Backend (api.py)
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      console.log("API Response:", data);

      // 3. Process Heatmap (Base64 to Image)
      if (data.heatmap_base64) {
        setGradcamImage(`data:image/png;base64,${data.heatmap_base64}`);
      }

      // 4. Process Prediction
      const confidencePercent = data.confidence * 100;
      
      const predictionResult: PredictionData = {
        prediction: data.class.toLowerCase().includes('malignant') ? 'malignant' : 'benign',
        confidence: confidencePercent,
      };
      setPrediction(predictionResult);

      // 5. Handle Yellow Flag & Notifications
      if (data.yellow_flag) {
        toast({
          title: "⚠️ Unsure Prediction (Yellow Flag)",
          description: "Confidence is near 50%. Please consult a radiologist.",
          variant: "destructive",
          duration: 6000,
        });
      } else {
        toast({
          title: t('common.success'),
          description: `${t('results.prediction')}: ${predictionResult.prediction === 'benign' ? t('results.benign') : t('results.malignant')}`,
        });
      }

    } catch (error) {
      console.error("Analysis Error:", error);
      toast({
        title: "Connection Error",
        description: "Could not connect to the Python server. Is api.py running on port 8000?",
        variant: 'destructive',
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="min-h-[calc(100vh-4rem)] p-4 md:p-6 lg:p-8">
      <div className={cn(
        "grid grid-cols-1 lg:grid-cols-12 gap-6 max-w-[1600px] mx-auto",
        isRTL && "lg:grid-flow-col-dense"
      )}>
        {/* Left Panel - Input & Controls */}
        <div className={cn("lg:col-span-4 space-y-6", isRTL && "lg:col-start-9")}>
          <Card className="shadow-soft">
            <CardHeader className="pb-3">
              <CardTitle className={cn("text-lg", isRTL && "text-right")}>
                {t('upload.title')}
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <ImageUpload
                onImageSelect={handleImageSelect}
                selectedImage={originalImage}
                onClear={handleClear}
              />
              
              <Separator />
              
              <AdvancedSettings
                settings={settings}
                onSettingsChange={setSettings}
              />
              
              <Separator />
              
              {/* Analyze Button */}
              <Button
                onClick={handleAnalyze}
                disabled={!originalImage || isAnalyzing}
                className={cn(
                  "w-full h-12 text-base font-semibold gap-2 bg-accent hover:bg-accent/90 text-accent-foreground",
                  isRTL && "flex-row-reverse"
                )}
              >
                {isAnalyzing ? (
                  <>
                    <Loader2 className="h-5 w-5 animate-spin" />
                    {t('analyze.analyzing')}
                  </>
                ) : (
                  <>
                    <Play className="h-5 w-5" />
                    {t('analyze.button')}
                  </>
                )}
              </Button>
            </CardContent>
          </Card>
        </div>

        {/* Right Panel - Visualization */}
        <div className={cn("lg:col-span-8 space-y-6", isRTL && "lg:col-start-1")}>
          <XAIViewer
            originalImage={originalImage}
            gradcamImage={gradcamImage}
            isLoading={isAnalyzing}
          />

          <div className={cn(
            "grid grid-cols-1 md:grid-cols-2 gap-6",
            isRTL && "md:grid-flow-col-dense"
          )}>
            <PredictionResult
              result={prediction}
              isLoading={isAnalyzing}
            />

            <Card className="shadow-soft">
              <CardHeader className="pb-3">
                <CardTitle className={cn("text-sm font-medium", isRTL && "text-right")}>
                  {t('results.export')}
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <ExportButton
                  originalImage={originalImage}
                  gradcamImage={gradcamImage}
                  prediction={prediction?.prediction || null}
                  confidence={prediction?.confidence || null}
                  disabled={isAnalyzing || !prediction}
                />
                
                {selectedFile && (
                  <ImageMetadata
                    filename={selectedFile.name}
                    dimensions={imageDimensions}
                    filesize={selectedFile.size}
                    modelUsed={prediction ? getModelName(settings.model) : null}
                  />
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;