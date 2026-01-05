import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useLanguage } from '@/contexts/LanguageContext';
import { Button } from '@/components/ui/button';
import { Microscope, LayoutDashboard, Users } from 'lucide-react';
import { cn } from '@/lib/utils';

const Navbar: React.FC = () => {
  const { language, setLanguage, t, isRTL } = useLanguage();
  const location = useLocation();

  const navLinks = [
    { path: '/', label: t('nav.dashboard'), icon: LayoutDashboard },
    { path: '/about', label: t('nav.about'), icon: Users },
  ];

  return (
    <header className="sticky top-0 z-50 w-full glass border-b border-border/50 shadow-soft">
      <div className="container flex h-16 items-center justify-between px-4 md:px-6">
        {/* Logo & Title */}
        <Link to="/" className="flex items-center gap-3 group">
          <div className="relative">
            <div className="absolute inset-0 bg-accent/30 rounded-xl blur-md group-hover:blur-lg transition-all" />
            <div className="relative bg-primary rounded-xl p-2">
              <Microscope className="h-6 w-6 text-primary-foreground" />
            </div>
          </div>
          <div className={cn("flex flex-col", isRTL && "text-right")}>
            <span className="font-bold text-lg text-foreground tracking-tight">
              {t('nav.title')}
            </span>
            <span className="text-xs text-muted-foreground hidden sm:block">
              {t('nav.subtitle')}
            </span>
          </div>
        </Link>

        {/* Navigation Links */}
        <nav className="flex items-center gap-1 md:gap-2">
          {navLinks.map((link) => {
            const isActive = location.pathname === link.path;
            const Icon = link.icon;
            
            return (
              <Link key={link.path} to={link.path}>
                <Button
                  variant={isActive ? "secondary" : "ghost"}
                  size="sm"
                  className={cn(
                    "gap-2 font-medium transition-all",
                    isActive && "bg-accent/20 text-accent-foreground shadow-sm",
                    isRTL && "flex-row-reverse"
                  )}
                >
                  <Icon className="h-4 w-4" />
                  <span className="hidden sm:inline">{link.label}</span>
                </Button>
              </Link>
            );
          })}

          {/* Language Toggle */}
          <div className="flex items-center gap-1 ml-2 md:ml-4 bg-secondary rounded-lg p-1">
            <Button
              variant={language === 'en' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setLanguage('en')}
              className={cn(
                "h-8 px-3 text-xs font-semibold transition-all",
                language === 'en' && "shadow-sm"
              )}
            >
              EN
            </Button>
            <Button
              variant={language === 'fa' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setLanguage('fa')}
              className={cn(
                "h-8 px-3 text-xs font-semibold font-persian transition-all",
                language === 'fa' && "shadow-sm"
              )}
            >
              فا
            </Button>
          </div>
        </nav>
      </div>
    </header>
  );
};

export default Navbar;