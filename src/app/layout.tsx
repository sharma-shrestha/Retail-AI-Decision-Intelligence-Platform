import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Toaster } from "@/components/ui/toaster";
import { QueryProvider } from "@/components/dashboard/query-provider";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Retail AI Decision Intelligence Platform",
  description: "AI-powered demand forecasting, SHAP explainability, inventory intelligence, and RAG-based copilot using M5 retail data.",
  keywords: ["retail", "AI", "forecasting", "SHAP", "inventory", "M5", "LightGBM", "CatBoost"],
  authors: [{ name: "Retail AI Platform" }],
  icons: {
    icon: "/logo.jpg",
  },
  openGraph: {
    title: "Retail AI Decision Intelligence Platform",
    description: "AI-powered demand forecasting, SHAP explainability, and inventory intelligence",
    siteName: "Retail AI Platform",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-background text-foreground`}
      >
        <QueryProvider>{children}</QueryProvider>
        <Toaster />
      </body>
    </html>
  );
}
