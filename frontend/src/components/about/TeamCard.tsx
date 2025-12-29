import React from 'react';
import { useLanguage } from '@/contexts/LanguageContext';
import { Card, CardContent } from '@/components/ui/card';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { cn } from '@/lib/utils';

interface TeamMember {
  name: string;
  role: string;
  bio: string;
  avatar?: string;
}

interface TeamCardProps {
  member: TeamMember;
}

const TeamCard: React.FC<TeamCardProps> = ({ member }) => {
  const { isRTL } = useLanguage();

  const getInitials = (name: string) => {
    return name
      .split(' ')
      .map(n => n[0])
      .join('')
      .toUpperCase()
      .slice(0, 2);
  };

  return (
    <Card className="group overflow-hidden shadow-soft hover:shadow-lg transition-all duration-300 hover:-translate-y-1">
      <CardContent className="p-6">
        <div className={cn("flex flex-col items-center text-center", isRTL && "text-content")}>
          {/* Avatar */}
          <div className="relative mb-4">
            <div className="absolute inset-0 bg-accent/30 rounded-full blur-xl opacity-0 group-hover:opacity-100 transition-opacity" />
            <Avatar className="h-24 w-24 border-4 border-background shadow-lg">
              <AvatarImage src={member.avatar} alt={member.name} />
              <AvatarFallback className="bg-primary text-primary-foreground text-xl font-bold">
                {getInitials(member.name)}
              </AvatarFallback>
            </Avatar>
          </div>

          {/* Info */}
          <h3 className="font-bold text-lg text-foreground mb-1">
            {member.name}
          </h3>
          <span className="inline-block px-3 py-1 rounded-full bg-accent/20 text-accent-foreground text-xs font-medium mb-3">
            {member.role}
          </span>
          <p className="text-sm text-muted-foreground leading-relaxed">
            {member.bio}
          </p>
        </div>
      </CardContent>
    </Card>
  );
};

export default TeamCard;