# 🎨 Breast Cancer Detection - Frontend Client

This directory contains the **Client-Side** application for the Breast Cancer Detection System. It is a modern, responsive Single Page Application (SPA) built with **React** and **Vite**, designed to provide a seamless user experience for uploading mammograms and visualizing diagnostic results.

The interface follows a clean, medical-grade aesthetic using **Shadcn UI** and **Tailwind CSS**.

## 🛠️ Tech Stack & Libraries

The frontend is built using a modern stack focused on performance, type safety, and component reusability:

### Core
* **Framework:** [React 18](https://react.dev/)
* **Build Tool:** [Vite](https://vitejs.dev/) (Fast HMR & Bundling)
* **Language:** [TypeScript](https://www.typescriptlang.org/) (Strict typing)

### UI & Styling
* **Styling Engine:** [Tailwind CSS](https://tailwindcss.com/)
* **Component Library:** [Shadcn UI](https://ui.shadcn.com/) (built on top of [Radix UI](https://www.radix-ui.com/))
* **Icons:** [Lucide React](https://lucide.dev/)
* **Animations:** `tailwindcss-animate`

### State & Logic
* **Routing:** [React Router DOM](https://reactrouter.com/)
* **Data Fetching:** [TanStack Query (React Query)](https://tanstack.com/query/latest) (For managing API state)
* **Form Handling:** [React Hook Form](https://react-hook-form.com/) + [Zod](https://zod.dev/) (Validation schema)
* **Charts:** [Recharts](https://recharts.org/) (For visualizing confidence scores)

## 🚀 Key Features

* **Interactive Dashboard:** A clean interface for managing patient scans.
* **Smart Upload:** Drag-and-drop functionality with client-side validation.
* **Result Visualization:**
    * Real-time rendering of **Grad-CAM Heatmaps**.
    * Confidence score gauges and probability charts.
* **Responsive Design:** Fully optimized for Desktop, Tablet, and Mobile views.
* **Toast Notifications:** Real-time feedback using `sonner`.

## 📂 Project Structure

```text
frontend/
├── public/              # Static assets (favicons, etc.)
├── src/
│   ├── components/      # Reusable UI components
│   │   ├── ui/          # Shadcn UI primitives (Button, Card, etc.)
│   │   ├── dashboard/   # Dashboard-specific widgets
│   │   └── layout/      # Navbar, Sidebar, Footer
│   ├── hooks/           # Custom React Hooks (use-toast, etc.)
│   ├── pages/           # Application Routes (Home, Dashboard, About)
│   ├── lib/             # Utilities (utils.ts)
│   ├── App.tsx          # Main Application Component
│   └── main.tsx         # Entry Point
├── package.json         # Dependencies & Scripts
├── tailwind.config.js   # Tailwind Configuration
└── vite.config.ts       # Vite Configuration
