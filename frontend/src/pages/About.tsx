import React from "react";
import { useLanguage } from "@/contexts/LanguageContext";
import { Card, CardContent } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import TeamCard from "@/components/about/TeamCard";
import MethodologyCard from "@/components/about/MethodologyCard";
import { Microscope, Contrast, Brain, Eye, GraduationCap } from "lucide-react";
import { cn } from "@/lib/utils";

const About: React.FC = () => {
  const { t, isRTL } = useLanguage();

  const methodologySteps = [
    {
      icon: Contrast,
      titleKey: "about.step1Title",
      descKey: "about.step1Desc",
      step: 1,
    },
    {
      icon: Brain,
      titleKey: "about.step2Title",
      descKey: "about.step2Desc",
      step: 2,
    },
    {
      icon: Eye,
      titleKey: "about.step3Title",
      descKey: "about.step3Desc",
      step: 3,
    },
  ];

  const teamMembers = [
    {
      name: isRTL ? "عضو تیم ۱" : "Elahe Moharrampour",
      image: "/image-data/member1.jpg",
      role: t("about.role.datascientist"),
      bio: isRTL
        ? "متخصص در یادگیری عمیق و پردازش تصویر پزشکی با تمرکز بر تشخیص سرطان."
        : "Specializing in deep learning and medical image processing with a focus on cancer detection.",
    },
    {
      name: isRTL ? "عضو تیم ۲" : "Hadi Goli Bidgoli",
      image: "/image-data/member2.jpg",
      role:  t("about.role.uiux"),
      bio: isRTL
        ? "طراحی تجربیات کاربری شهودی و قابل دسترس برای ابزارهای تحقیقاتی پزشکی."
        : "Designing intuitive and accessible user experiences for medical research tools.",
    },
    {
      name: isRTL ? "عضو تیم ۳" : "Alireza Shams",
      image: "/image-data/member3.jpg",
      role: t("about.role.mleng"),
      bio: isRTL
        ? "متمرکز بر بهینه‌سازی مدل و استقرار برای کاربردهای بلادرنگ در زمینه پزشکی."
        : "Focused on model optimization and deployment for real-time medical applications.",
    },
  ];

  return (
    <div className="min-h-[calc(100vh-4rem)] p-4 md:p-6 lg:p-8">
      <div className="max-w-6xl space-y-12 mx-auto">
        {/* Hero Section */}
        <section
          className={cn("text-center space-y-6", isRTL && "text-content")}
        >
          <div className="inline-flex items-center justify-center gap-3">
            <div className="relative">
              <div className="absolute inset-0 bg-accent/30 rounded-2xl blur-xl" />
              <div className="relative bg-primary rounded-2xl p-4">
                <Microscope className="h-10 w-10 text-primary-foreground" />
              </div>
            </div>
          </div>

          <div className="space-y-3">
            <h1 className="text-3xl md:text-4xl font-bold text-foreground">
              {t("about.projectTitle")}
            </h1>
            <p className="text-lg text-muted-foreground max-w-3xl mx-auto leading-relaxed">
              {t("about.projectDesc")}
            </p>
          </div>

          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-secondary/80 text-sm text-muted-foreground">
            <GraduationCap className="h-4 w-4" />
            {t("about.university")}
          </div>
        </section>

               <Separator />
        {/* Project Goals and Motivation */}
        <section className="space-y-8">
          <div className={cn("text-center space-y-3", isRTL && "text-content")}>
            <h2 className="text-2xl md:text-3xl font-bold text-foreground">
              Project Goals and Motivation
            </h2>
            <p className=" max-w-2xl mx-auto">
              The goal of this project is early breast cancer detection using
              mammography images. The system acts as a decision-support tool for
              radiologists, focusing on subtle calcification patterns that may
              indicate early-stage cancer. It was developed by undergraduate
              students at Kharazmi University as part of the Artificial
              Intelligence course in Fall 2025, under the supervision of Dr.
              Bolhasani.
            </p>
          </div>
        </section>

        <Separator />

        {/* Methodology Section */}
        <section className="space-y-8">
          <div className={cn("text-center space-y-3", isRTL && "text-content")}>
            <h2 className="text-2xl md:text-3xl font-bold text-foreground">
              {t("about.methodology")}
            </h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              {t("about.methodologyDesc")}
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {methodologySteps.map((step, index) => (
              <MethodologyCard key={index} step={step} />
            ))}
          </div>
        </section>

        <Separator />

        {/* Team Section */}
        <section className="space-y-8">
          <div className={cn("text-center space-y-3", isRTL && "text-content")}>
            <h2 className="text-2xl md:text-3xl font-bold text-foreground">
              {t("about.team")}
            </h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              {t("about.teamDesc")}
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {teamMembers.map((member, index) => (
              <TeamCard key={index} member={member} />
            ))}
          </div>
        </section>
      

        {/* Footer */}
        <Card className="bg-gradient-to-r from-primary/5 to-accent/10 border-none shadow-soft">
          <CardContent
            className={cn("py-8 text-center", isRTL && "text-content")}
          >
            <p className="text-muted-foreground">
              © {new Date().getFullYear()} MicroCalc Research Project
            </p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default About;
