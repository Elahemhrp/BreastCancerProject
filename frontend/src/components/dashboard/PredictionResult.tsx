import React from 'react';
import { useLanguage } from '@/contexts/LanguageContext';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { AlertTriangle, CheckCircle2, XCircle, AlertCircle } from 'lucide-react';
import { cn } from '@/lib/utils';

export interface PredictionData {
  prediction: 'benign' | 'malignant';
  confidence: number;
}

interface PredictionResultProps {
  result: PredictionData | null;
  isLoading?: boolean;
}

const PredictionResult: React.FC<PredictionResultProps> = ({ result, isLoading }) => {
  const { t, isRTL } = useLanguage();

  const isUncertain = result && result.confidence >= 45 && result.confidence <= 55;

  const getStatusConfig = () => {
    if (!result) return null;
    
    if (isUncertain) {
      return {
        icon: AlertCircle,
        label: t('results.uncertain'),
        badgeClass: 'badge-uncertain',
        color: 'text-warning',
        bgColor: 'bg-warning/10',
      };
    }
    
    if (result.prediction === 'benign') {
      return {
        icon: CheckCircle2,
        label: t('results.benign'),
        badgeClass: 'badge-benign',
        color: 'text-success',
        bgColor: 'bg-success/10',
      };
    }
    
    return {
      icon: XCircle,
      label: t('results.malignant'),
      badgeClass: 'badge-malignant',
      color: 'text-destructive',
      bgColor: 'bg-destructive/10',
    };
  };

  if (isLoading) {
    return (
      <Card className="shadow-soft">
        <CardContent className="p-6">
          <div className="flex flex-col items-center justify-center py-8">
            <div className="relative">
              <div className="absolute inset-0 bg-accent/30 rounded-full blur-xl animate-pulse-soft" />
              <div className="relative h-16 w-16 rounded-full border-4 border-accent/30 border-t-accent animate-spin-slow" />
            </div>
            <p className="mt-4 text-sm text-muted-foreground">{t('analyze.analyzing')}</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!result) {
    return (
      <Card className="shadow-soft">
        <CardContent className="p-6">
          <div className="flex flex-col items-center justify-center py-8 text-center">
            <div className="bg-secondary rounded-full p-4 mb-4">
              <AlertTriangle className="h-8 w-8 text-muted-foreground" />
            </div>
            <p className="text-sm text-muted-foreground">{t('results.noResults')}</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  const status = getStatusConfig()!;
  const StatusIcon = status.icon;

  return (
    <Card className={cn("shadow-soft overflow-hidden animate-fade-in", status.bgColor)}>
      <CardHeader className="pb-2">
        <CardTitle className={cn("text-sm font-medium", isRTL && "text-right")}>
          {t('results.title')}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Prediction Badge */}
        <div className={cn("flex items-center gap-3", isRTL && "flex-row-reverse justify-end")}>
          <StatusIcon className={cn("h-8 w-8", status.color)} />
          <div className={cn("space-y-1", isRTL && "text-right")}>
            <p className="text-xs text-muted-foreground uppercase tracking-wide">
              {t('results.prediction')}
            </p>
            <Badge className={cn("text-base px-3 py-1", status.badgeClass)}>
              {status.label}
            </Badge>
          </div>
        </div>

        {/* Confidence Score */}
        <div className="space-y-2">
          <div className={cn("flex items-center justify-between", isRTL && "flex-row-reverse")}>
            <span className="text-sm text-muted-foreground">{t('results.confidence')}</span>
            <span className={cn("text-lg font-bold", status.color)}>
              {result.confidence.toFixed(1)}%
            </span>
          </div>
          <Progress 
            value={result.confidence} 
            className="h-2"
          />
        </div>

        {/* Uncertain Warning */}
        {isUncertain && (
          <div className="mt-4 p-3 rounded-lg bg-warning/20 border border-warning/30 animate-fade-in">
            <div className={cn("flex items-start gap-2", isRTL && "flex-row-reverse")}>
              <AlertTriangle className="h-5 w-5 text-warning mt-0.5 flex-shrink-0" />
              <div className={cn("space-y-1", isRTL && "text-right")}>
                <p className="font-medium text-sm text-warning-foreground">
                  {t('results.warning')}
                </p>
                <p className="text-xs text-warning-foreground/80">
                  {t('results.warningDesc')}
                </p>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default PredictionResult;