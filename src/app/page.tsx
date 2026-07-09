"use client";

import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import {
  LayoutDashboard, TrendingUp, Brain, Package, BarChart3, Bot,
} from "lucide-react";
import { OverviewTab } from "@/components/dashboard/overview-tab";
import { ForecastTab } from "@/components/dashboard/forecast-tab";
import { ExplainTab } from "@/components/dashboard/explain-tab";
import { InventoryTab } from "@/components/dashboard/inventory-tab";
import { AnalyticsTab } from "@/components/dashboard/analytics-tab";
import { CopilotTab } from "@/components/dashboard/copilot-tab";

const TABS = [
  { value: "overview",  label: "Overview",   icon: LayoutDashboard },
  { value: "forecast",  label: "Forecast",   icon: TrendingUp },
  { value: "explain",   label: "Explain",    icon: Brain },
  { value: "inventory", label: "Inventory",  icon: Package },
  { value: "analytics", label: "Analytics",  icon: BarChart3 },
  { value: "copilot",   label: "AI Copilot", icon: Bot },
];

export default function DashboardPage() {
  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b bg-card">
        <div className="container mx-auto flex items-center justify-between px-4 py-3">
          <div>
            <h1 className="text-xl font-bold tracking-tight">
              Retail AI Platform
            </h1>
            <p className="text-xs text-muted-foreground">
              M5 Forecasting Dataset · 3 Models · SHAP Explainability · RAG Copilot
            </p>
          </div>
          <div className="text-right text-xs text-muted-foreground hidden sm:block">
            <div>CatBoost (Best) · LightGBM · XGBoost</div>
            <div>80 engineered features · 99 products · 10 stores</div>
          </div>
        </div>
      </header>

      {/* Tabs */}
      <div className="container mx-auto px-4 py-4">
        <Tabs defaultValue="overview" className="w-full">
          <TabsList className="w-full justify-start overflow-x-auto mb-6 h-auto p-1">
            {TABS.map(({ value, label, icon: Icon }) => (
              <TabsTrigger key={value} value={value} className="gap-1.5 px-3 text-xs sm:text-sm">
                <Icon className="h-4 w-4" />
                <span className="hidden sm:inline">{label}</span>
              </TabsTrigger>
            ))}
          </TabsList>

          <TabsContent value="overview"><OverviewTab /></TabsContent>
          <TabsContent value="forecast"><ForecastTab /></TabsContent>
          <TabsContent value="explain"><ExplainTab /></TabsContent>
          <TabsContent value="inventory"><InventoryTab /></TabsContent>
          <TabsContent value="analytics"><AnalyticsTab /></TabsContent>
          <TabsContent value="copilot"><CopilotTab /></TabsContent>
        </Tabs>
      </div>
    </div>
  );
}