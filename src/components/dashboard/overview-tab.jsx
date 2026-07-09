"use client";

import { useQuery } from "@tanstack/react-query";
import { fetchAnalyticsSummary, fetchModels } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  ChartContainer, ChartTooltip, ChartTooltipContent, ChartLegend, ChartLegendContent,
} from "@/components/ui/chart";
import { Bar, BarChart, XAxis, YAxis, CartesianGrid } from "recharts";
import { Package, Store, Trophy, Database } from "lucide-react";
import { motion } from "framer-motion";

const kpiIcon = { Package, Store, Trophy, Database };

function KpiCard({ icon, label, value, delay }) {
  const Icon = kpiIcon[icon];
  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay }}>
      <Card>
        <CardContent className="flex items-center gap-4">
          <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-full bg-muted">
            <Icon className="h-6 w-6 text-muted-foreground" />
          </div>
          <div>
            <p className="text-2xl font-bold">{value}</p>
            <p className="text-sm text-muted-foreground">{label}</p>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}

function Pulse() {
  return (
    <Card>
      <CardContent className="h-[100px] animate-pulse bg-muted/50 rounded-lg" />
    </Card>
  );
}

export function OverviewTab() {
  const { data: summary, isLoading: sumLoading, error: sumError } = useQuery({
    queryKey: ["analytics-summary"],
    queryFn: fetchAnalyticsSummary,
  });

  const { data: modelsData } = useQuery({
    queryKey: ["models"],
    queryFn: fetchModels,
  });

  if (sumError) {
    return (
      <Card className="border-destructive">
        <CardContent className="py-8 text-center">
          <p className="text-destructive font-medium">Failed to load overview</p>
          <p className="text-sm text-muted-foreground mt-1">Is the FastAPI backend running on port 8000?</p>
        </CardContent>
      </Card>
    );
  }

  const ov = summary?.overview;
  const mp = summary?.model_performance || modelsData?.models;

  // Build chart data — handles both flat {rmse,mae,rmsse} and nested {metrics:{...}}
  const getMetric = (info, key) => info.metrics ? info.metrics[key] : info[key];

  const chartData = [];
  if (mp) {
    ["rmse", "mae", "rmsse"].forEach((metric) => {
      const row = { metric: metric.toUpperCase() };
      Object.entries(mp).forEach(([name, info]) => {
        const val = getMetric(info, metric);
        if (val !== undefined) row[name] = val;
      });
      chartData.push(row);
    });
  }

  // Find best model
  let bestModel = "—";
  let bestRmse = Infinity;
  if (mp) {
    Object.entries(mp).forEach(([name, info]) => {
      const rmse = getMetric(info, "rmse");
      if (rmse !== undefined && rmse < bestRmse) {
        bestRmse = rmse;
        bestModel = name.charAt(0).toUpperCase() + name.slice(1);
      }
    });
  }

  const chartConfig = {
    catboost: { label: "CatBoost", color: "hsl(var(--chart-1))" },
    lightgbm: { label: "LightGBM", color: "hsl(var(--chart-2))" },
    xgboost: { label: "XGBoost", color: "hsl(var(--chart-3))" },
  };

  return (
    <div className="space-y-6">
      {/* KPI Grid */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {sumLoading ? (
          <><Pulse /><Pulse /><Pulse /><Pulse /></>
        ) : (
          <>
            <KpiCard icon="Package" label="Unique Products" value={ov?.unique_products?.toLocaleString() || "—"} delay={0} />
            <KpiCard icon="Store" label="Stores" value={ov?.unique_stores || "—"} delay={0.1} />
            <KpiCard icon="Trophy" label={`Best Model (${bestModel})`} value={bestRmse < Infinity ? bestRmse.toFixed(4) : "—"} delay={0.2} />
            <KpiCard icon="Database" label="Total Data Points" value={ov?.total_data_points?.toLocaleString() || "—"} delay={0.3} />
          </>
        )}
      </div>

      {/* Extra Info Row */}
      {!sumLoading && ov && (
        <div className="flex flex-wrap gap-4 text-sm text-muted-foreground">
          <span>Total Sales: <strong className="text-foreground">{ov.total_sales?.toLocaleString()}</strong></span>
          <span>Revenue: <strong className="text-foreground">${ov.total_revenue?.toLocaleString()}</strong></span>
          <span>Avg Daily: <strong className="text-foreground">{ov.avg_daily_sales}</strong></span>
          <span>Zero Sales: <strong className="text-foreground">{ov.zero_sales_pct}%</strong></span>
          <span>Date Range: <strong className="text-foreground">{ov.date_range}</strong></span>
        </div>
      )}

      {/* Model Comparison Chart */}
      <Card>
        <CardHeader>
          <CardTitle>Model Performance Comparison</CardTitle>
        </CardHeader>
        <CardContent>
          {chartData.length === 0 ? (
            <div className="h-[300px] animate-pulse bg-muted/30 rounded-lg" />
          ) : (
            <ChartContainer config={chartConfig} className="h-[300px] w-full">
              <BarChart data={chartData} layout="vertical" margin={{ left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                <YAxis dataKey="metric" type="category" width={60} />
                <XAxis type="number" />
                <ChartTooltip content={<ChartTooltipContent />} />
                <ChartLegend content={<ChartLegendContent />} />
                <Bar dataKey="catboost" fill="var(--color-catboost)" radius={[0, 4, 4, 0]} />
                <Bar dataKey="lightgbm" fill="var(--color-lightgbm)" radius={[0, 4, 4, 0]} />
                <Bar dataKey="xgboost" fill="var(--color-xgboost)" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ChartContainer>
          )}
        </CardContent>
      </Card>
    </div>
  );
}