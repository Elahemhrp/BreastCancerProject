import React from 'react';
import { useLanguage } from '@/contexts/LanguageContext';
import { Settings, ChevronDown } from 'lucide-react';
import { Switch } from '@/components/ui/switch';
import { Slider } from '@/components/ui/slider';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible';
import { cn } from '@/lib/utils';

export interface AnalysisSettings {
  claheEnabled: boolean;
  model: 'efficientnet' | 'resnet' | 'custom';
  threshold: number;
}

interface AdvancedSettingsProps {
  settings: AnalysisSettings;
  onSettingsChange: (settings: AnalysisSettings) => void;
}

const AdvancedSettings: React.FC<AdvancedSettingsProps> = ({ settings, onSettingsChange }) => {
  const { t, isRTL } = useLanguage();
  const [isOpen, setIsOpen] = React.useState(false);

  const updateSetting = <K extends keyof AnalysisSettings>(
    key: K,
    value: AnalysisSettings[K]
  ) => {
    onSettingsChange({ ...settings, [key]: value });
  };

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen} className="w-full">
      <CollapsibleTrigger className={cn(
        "flex items-center justify-between w-full p-3 rounded-lg bg-secondary/50 hover:bg-secondary transition-colors group",
        isRTL && "flex-row-reverse"
      )}>
        <div className={cn("flex items-center gap-2", isRTL && "flex-row-reverse")}>
          <Settings className="h-4 w-4 text-muted-foreground" />
          <span className="font-medium text-sm">{t('settings.title')}</span>
        </div>
        <ChevronDown className={cn(
          "h-4 w-4 text-muted-foreground transition-transform duration-200",
          isOpen && "rotate-180"
        )} />
      </CollapsibleTrigger>

      <CollapsibleContent className="mt-3 space-y-4 animate-accordion-down">
        {/* CLAHE Toggle */}
        <div className={cn(
          "flex items-center justify-between p-3 rounded-lg bg-card border border-border",
          isRTL && "flex-row-reverse"
        )}>
          <div className={cn("space-y-0.5", isRTL && "text-right")}>
            <Label className="text-sm font-medium">{t('settings.clahe')}</Label>
            <p className="text-xs text-muted-foreground">{t('settings.claheDesc')}</p>
          </div>
          <Switch
            checked={settings.claheEnabled}
            onCheckedChange={(checked) => updateSetting('claheEnabled', checked)}
          />
        </div>

        {/* Model Selection */}
        <div className={cn("space-y-2", isRTL && "text-right")}>
          <Label className="text-sm font-medium">{t('settings.model')}</Label>
          <p className="text-xs text-muted-foreground">{t('settings.modelDesc')}</p>
          <Select
            value={settings.model}
            onValueChange={(value) => updateSetting('model', value as AnalysisSettings['model'])}
          >
            <SelectTrigger className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="efficientnet">{t('model.efficientnet')}</SelectItem>
              <SelectItem value="resnet">{t('model.resnet')}</SelectItem>
              <SelectItem value="custom">{t('model.custom')}</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Threshold Slider */}
        <div className={cn("space-y-3", isRTL && "text-right")}>
          <div className={cn("flex items-center justify-between", isRTL && "flex-row-reverse")}>
            <div>
              <Label className="text-sm font-medium">{t('settings.threshold')}</Label>
              <p className="text-xs text-muted-foreground">{t('settings.thresholdDesc')}</p>
            </div>
            <span className="text-sm font-mono bg-secondary px-2 py-1 rounded">
              {settings.threshold}%
            </span>
          </div>
          <Slider
            value={[settings.threshold]}
            onValueChange={([value]) => updateSetting('threshold', value)}
            min={0}
            max={100}
            step={5}
            className="w-full"
          />
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
};

export default AdvancedSettings;