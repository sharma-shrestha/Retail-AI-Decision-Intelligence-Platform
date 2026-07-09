"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchTopProducts, fetchFeatureImportance, fetchAnalyticsSummary } from "@/lib/api";
import { MODEL_OPTIONS } from "@/lib/constants";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import {
  ChartContainer, ChartTooltip, ChartTooltipContent,
} from "@/components/ui/chart";
import { Bar, BarChart, XAxis, YAxis, CartesianGrid } from "recharts";
import { BarChart3, Trophy, Loader2 } from "lucide-react";
import { motion } from "framer-motion";

export function AnalyticsTab() {
  const [metric, setMetric] = useState("total_sales");
  const [topN, setTopN] = useState("10");
  const [fiModel, setFiModel] = useState("catboost");
  const [fiEnabled, setFiEnabled] = useState(false);

  const { data: topData, isLoading: topLoading } = useQuery({
    queryKey: ["top-products-analytics", topN, metric],
    queryFn: () => fetchTopProducts(parseInt(topN), metric),
  });

  const { data: fiData, isLoading: fiLoading } = useQuery({
    queryKey: ["feature-importance", fiModel],
    queryFn: () => fetchFeatureImportance(fiModel, 20),
    enabled: fiEnabled,
    staleTime: 300000, // 5 min — SHAP computation is expensive
  });

  const { data: summary } = useQuery({
    queryKey: ["analytics-summary-store"],
    queryFn: fetchAnalyticsSummary,
  });

  const metricLabel = { total_sales: "Total Sales", avg_daily: "Avg Daily Sales", revenue: "Revenue" };
  const valueKey = { total_sales: "total_sales", avg_daily: "avg_daily_sales", revenue: "total_revenue" };

  // Feature importance chart data — API returns mean_abs_shap, map to "importance" for Recharts
  const fiChartData = (fiData?.top_features || []).slice().reverse().map((f) => ({
    ...f,
    importance: f.mean_abs_shap || f.importance || 0,
  }));

  return (
    <div className="space-y-6">
      {/* Top Products */}
      <Card>
        <CardHeader>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <CardTitle className="flex items-center gap-2"><Trophy className="h-5 w-5" /> Top Products</CardTitle>
            <div className="flex gap-2">
              <Select value={metric} onValueChange={setMetric}>
                <SelectTrigger className="w-[150px]"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="total_sales">Total Sales</SelectItem>
                  <SelectItem value="avg_daily">Avg Daily Sales</SelectItem>
                  <SelectItem value="revenue">Revenue</SelectItem>
                </SelectContent>
              </Select>
              <Select value={topN} onValueChange={setTopN}>
                <SelectTrigger className="w-[80px]"><SelectValue /></SelectTrigger>
                <SelectContent>
                  {[10, 20, 50].map((n) => <SelectItem key={n} value={String(n)}>{n}</SelectItem>)}
                </SelectContent>
              </Select>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {topLoading ? (
            <div className="h-[300px] animate-pulse bg-muted/30 rounded-lg" />
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-[50px]">#</TableHead>
                  <TableHead>Item ID</TableHead>
                  <TableHead>Store</TableHead>
                  <TableHead className="text-right">{metricLabel[metric]}</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {topData?.products?.map((p, i) => (
                  <TableRow key={i}>
                    <TableCell className="font-mono text-muted-foreground">{i + 1}</TableCell>
                    <TableCell className="font-medium">{p.item_id}</TableCell>
                    <TableCell><Badge variant="outline">{p.store_id}</Badge></TableCell>
                    <TableCell className="text-right font-mono font-medium">
                      {(p[valueKey[metric]] || 0).toLocaleString(undefined, { maximumFractionDigits: 2 })}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        {/* Feature Importance */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center gap-2"><BarChart3 className="h-5 w-5" /> Feature Importance</CardTitle>
              <div className="flex items-center gap-2">
                <Select value={fiModel} onValueChange={(v) => { setFiModel(v); setFiEnabled(false); }}>
                  <SelectTrigger className="w-[130px] h-8"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {MODEL_OPTIONS.map((m) => <SelectItem key={m.value} value={m.value}>{m.label}</SelectItem>)}
                  </SelectContent>
                </Select>
                <Button size="sm" variant="outline" onClick={() => setFiEnabled(true)} disabled={fiLoading}>
                  {fiLoading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <BarChart3 className="h-3.5 w-3.5" />}
                  {fiData ? "Refresh" : "Load"}
                </Button>
              </div>
            </div>
            <CardDescription>Top 20 features by SHAP importance{fiData ? (" (" + fiData.total_features + " total)") : ""}</CardDescription>
          </CardHeader>
          <CardContent>
            {!fiEnabled && !fiData ? (
              <div className="h-[400px] flex items-center justify-center text-muted-foreground text-sm">
                Click "Load" to compute SHAP feature importance
              </div>
            ) : fiLoading ? (
              <div className="h-[400px] animate-pulse bg-muted/30 rounded-lg" />
            ) : (
              <ChartContainer config={{ importance: { label: "Importance" } }} className="h-[400px] w-full">
                <BarChart data={fiChartData} layout="vertical" margin={{ left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                  <XAxis type="number" />
                  <YAxis dataKey="feature" type="category" width={140} tick={{ fontSize: 10 }} />
                  <ChartTooltip content={<ChartTooltipContent />} />
                  <Bar dataKey="importance" fill="hsl(var(--chart-1))" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ChartContainer>
            )}
          </CardContent>
        </Card>

        {/* Store Performance */}
        <Card>
          <CardHeader><CardTitle>Store Performance</CardTitle></CardHeader>
          <CardContent>
            {summary?.by_store ? (
              <div className="grid gap-3 sm:grid-cols-2">
                {Object.entries(summary.by_store).map(([store, s]) => (
                  <motion.div key={store} initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                    <div className="rounded-lg border p-3 space-y-1">
                      <div className="flex items-center justify-between">
                        <span className="font-semibold">{store}</span>
                        <Badge variant="outline">{s.n_products} products</Badge>
                      </div>
                      <div className="text-sm text-muted-foreground space-y-0.5">
                        <div className="flex justify-between"><span>Total Sales</span><span className="font-medium text-foreground">{s.total_sales?.toLocaleString()}</span></div>
                        <div className="flex justify-between"><span>Avg Daily</span><span className="font-medium text-foreground">{s.avg_daily_sales}</span></div>
                        <div className="flex justify-between"><span>Avg Price</span><span className="font-medium text-foreground">${s.avg_price}</span></div>
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            ) : (
              <div className="h-[300px] animate-pulse bg-muted/30 rounded-lg" />
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}