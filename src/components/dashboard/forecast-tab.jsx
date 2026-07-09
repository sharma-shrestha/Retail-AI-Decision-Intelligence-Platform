"use client";

import { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchForecast, fetchTopProducts } from "@/lib/api";
import { DataSelectors } from "./data-selectors";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
  ChartContainer, ChartTooltip, ChartTooltipContent,
} from "@/components/ui/chart";
import { Bar, BarChart, XAxis, YAxis, CartesianGrid, Cell, ReferenceLine } from "recharts";
import { TrendingUp, Zap, Clock, BarChart3, Loader2 } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

export function ForecastTab() {
  const [storeId, setStoreId] = useState("CA_1");
  const [model, setModel] = useState("catboost");
  const [days, setDays] = useState("28");
  const [forecastKey, setForecastKey] = useState(0);
  const [selectedItem, setSelectedItem] = useState("");

  // Fetch available items
  const { data: topData, isLoading: itemsLoading } = useQuery({
    queryKey: ["top-products", 50, "total_sales"],
    queryFn: () => fetchTopProducts(50, "total_sales"),
  });

  const items = useMemo(() => {
    if (!topData?.products) return [];
    const unique = [...new Set(topData.products.map((p) => p.item_id))];
    return unique;
  }, [topData]);

  // Set first item when loaded
  useMemo(() => {
    if (items.length > 0 && !selectedItem) setSelectedItem(items[0]);
  }, [items]);

  // Fetch forecast (manual trigger)
  const { data: forecast, isLoading: fcLoading, error: fcError } = useQuery({
    queryKey: ["forecast", storeId, selectedItem, model, days, forecastKey],
    queryFn: () => fetchForecast({
      store_id: storeId, item_id: selectedItem, model, days: parseInt(days),
    }),
    enabled: !!selectedItem && forecastKey > 0,
  });

  const handleGenerate = () => {
    if (selectedItem) setForecastKey((k) => k + 1);
  };

  const chartData = forecast ? [
    { name: "Hist. Mean", value: forecast.historical_context.mean_daily_sales, color: "hsl(var(--chart-2))" },
    { name: "Predicted", value: forecast.forecast.predicted_daily_sales, color: "hsl(var(--chart-1))" },
    { name: "Hist. Max", value: forecast.historical_context.max_daily_sales, color: "hsl(var(--chart-4))" },
  ] : [];

  return (
    <div className="space-y-6">
      {/* Controls */}
      <Card>
        <CardContent className="pt-6 space-y-4">
          <div className="flex flex-wrap items-end gap-3">
            <DataSelectors
              storeId={storeId} itemId={selectedItem} model={model}
              onStoreChange={setStoreId} onItemChange={setSelectedItem} onModelChange={setModel}
              items={items} itemsLoading={itemsLoading}
            />
            <div className="flex flex-col gap-1.5">
              <label className="text-xs font-medium text-muted-foreground">Days</label>
              <Select value={days} onValueChange={setDays}>
                <SelectTrigger className="w-[80px]"><SelectValue /></SelectTrigger>
                <SelectContent>
                  {[7, 14, 21, 28].map((d) => (
                    <SelectItem key={d} value={String(d)}>{d}d</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <Button onClick={handleGenerate} disabled={!selectedItem || fcLoading} className="ml-2">
              {fcLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <TrendingUp className="h-4 w-4" />}
              Generate Forecast
            </Button>
          </div>
        </CardContent>
      </Card>

      {fcError && (
        <Card className="border-destructive">
          <CardContent className="py-4 text-center text-destructive text-sm">{fcError.message}</CardContent>
        </Card>
      )}

      <AnimatePresence>
        {forecast && (
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="space-y-4">
            {/* Main Forecast */}
            <div className="grid gap-4 md:grid-cols-3">
              <Card>
                <CardHeader className="pb-2">
                  <CardDescription>Daily Prediction</CardDescription>
                  <CardTitle className="text-3xl">{forecast.forecast.predicted_daily_sales}</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">units/day for {selectedItem} at {storeId}</p>
                  <Badge variant="secondary" className="mt-2">{model.toUpperCase()}</Badge>
                  <Badge variant="outline" className="ml-2">{forecast.latency_ms}ms</Badge>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-2">
                  <CardDescription>Period Forecast ({days} days)</CardDescription>
                  <CardTitle className="text-3xl">{forecast.forecast.predicted_period_sales}</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">total predicted units</p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-2">
                  <CardDescription>Model Accuracy (RMSE)</CardDescription>
                  <CardTitle className="text-3xl">{forecast.model_metrics.rmse}</CardTitle>
                </CardHeader>
                <CardContent className="space-y-1 text-sm text-muted-foreground">
                  <div className="flex justify-between"><span>MAE</span><span className="font-medium text-foreground">{forecast.model_metrics.mae}</span></div>
                  <div className="flex justify-between"><span>RMSSE</span><span className="font-medium text-foreground">{forecast.model_metrics.rmsse}</span></div>
                </CardContent>
              </Card>
            </div>

            {/* Historical Context + Chart */}
            <div className="grid gap-4 md:grid-cols-2">
              <Card>
                <CardHeader><CardTitle>Historical Context</CardTitle></CardHeader>
                <CardContent className="space-y-3">
                  {[
                    { icon: BarChart3, label: "Mean Daily Sales", value: forecast.historical_context.mean_daily_sales },
                    { icon: TrendingUp, label: "Std Deviation", value: forecast.historical_context.std_daily_sales },
                    { icon: Zap, label: "Max Daily Sales", value: forecast.historical_context.max_daily_sales },
                    { icon: Clock, label: "Data Points", value: forecast.historical_context.data_points },
                  ].map(({ icon: I, label, value }) => (
                    <div key={label} className="flex items-center justify-between">
                      <div className="flex items-center gap-2 text-sm text-muted-foreground">
                        <I className="h-4 w-4" />{label}
                      </div>
                      <span className="font-medium">{value}</span>
                    </div>
                  ))}
                </CardContent>
              </Card>

              <Card>
                <CardHeader><CardTitle>Forecast vs History</CardTitle></CardHeader>
                <CardContent>
                  <ChartContainer config={{ value: { label: "Sales" } }} className="h-[200px] w-full">
                    <BarChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis />
                      <ChartTooltip content={<ChartTooltipContent />} />
                      <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                        {chartData.map((d, i) => <Cell key={i} fill={d.color} />)}
                      </Bar>
                    </BarChart>
                  </ChartContainer>
                </CardContent>
              </Card>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}