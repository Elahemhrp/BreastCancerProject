import React from 'react';
import { useLanguage } from '@/contexts/LanguageContext';
import { Card, CardContent } from '@/components/ui/card';
import { LucideIcon } from 'lucide-react';
import { cn } from '@/lib/utils';

interface MethodologyStep {
  icon: LucideIcon;
  titleKey: string;
  descKey: string;
  step: number;
}

interface MethodologyCardProps {
  step: MethodologyStep;
}

const MethodologyCard: React.FC<MethodologyCardProps> = ({ step }) => {
  const { t, isRTL } = useLanguage();
  const Icon = step.icon;

  return (
    <Card className="group overflow-hidden shadow-soft hover:shadow-lg transition-all duration-300 border-t-4 border-t-accent">
      <CardContent className="p-6">
        <div className={cn("space-y-4", isRTL && "text-right")}>
          {/* Step Number & Icon */}
          <div className={cn("flex items-center gap-4", isRTL && "flex-row-reverse")}>
            <div className="relative">
              <div className="absolute inset-0 bg-accent/20 rounded-xl blur-md group-hover:blur-lg transition-all" />
              <div className="relative flex items-center justify-center h-14 w-14 rounded-xl bg-primary text-primary-foreground">
                <Icon className="h-6 w-6" />
              </div>
            </div>
            <div className="flex items-center justify-center h-8 w-8 rounded-full bg-secondary text-foreground font-bold text-sm">
              {step.step}
            </div>
          </div>

          {/* Content */}
          <div>
            <h3 className="font-bold text-lg text-foreground mb-2">
              {t(step.titleKey)}
            </h3>
            <p className="text-sm text-muted-foreground leading-relaxed">
              {t(step.descKey)}
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default MethodologyCard;