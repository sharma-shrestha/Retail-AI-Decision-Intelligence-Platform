"use client";

import { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchExplain, fetchTopProducts } from "@/lib/api";
import { DataSelectors } from "./data-selectors";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  ChartContainer, ChartTooltip, ChartTooltipContent,
} from "@/components/ui/chart";
import { Bar, BarChart, XAxis, YAxis, CartesianGrid, Cell, ReferenceLine } from "recharts";
import { Brain, Loader2, ArrowUp, ArrowDown, Info } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

export function ExplainTab() {
  const [storeId, setStoreId] = useState("CA_1");
  const [model, setModel] = useState("catboost");
  const [explainKey, setExplainKey] = useState(0);
  const [selectedItem, setSelectedItem] = useState("");

  const { data: topData, isLoading: itemsLoading } = useQuery({
    queryKey: ["top-products-explain"],
    queryFn: () => fetchTopProducts(50, "total_sales"),
  });

  const items = useMemo(() => {
    if (!topData?.products) return [];
    return [...new Set(topData.products.map((p) => p.item_id))];
  }, [topData]);

  useMemo(() => {
    if (items.length > 0 && !selectedItem) setSelectedItem(items[0]);
  }, [items]);

  const { data: explain, isLoading: exLoading, error: exError } = useQuery({
    queryKey: ["explain", storeId, selectedItem, model, explainKey],
    queryFn: () => fetchExplain({ store_id: storeId, item_id: selectedItem, model }),
    enabled: !!selectedItem && explainKey > 0,
  });

  // Build waterfall chart data
  const chartData = useMemo(() => {
    if (!explain?.explanation) return [];
    const pos = (explain.explanation.top_positive_contributors || []).slice(0, 5).map((c) => ({
      feature: c.feature.length > 25 ? c.feature.slice(0, 22) + "..." : c.feature,
      impact: c.impact, type: "positive",
    }));
    const neg = (explain.explanation.top_negative_contributors || []).slice(0, 5).map((c) => ({
      feature: c.feature.length > 25 ? c.feature.slice(0, 22) + "..." : c.feature,
      impact: c.impact, type: "negative",
    }));
    return [...pos.reverse(), ...neg].sort((a, b) => a.impact - b.impact);
  }, [explain]);

  return (
    <div className="space-y-6">
      <Card>
        <CardContent className="pt-6">
          <div className="flex flex-wrap items-end gap-3">
            <DataSelectors
              storeId={storeId} itemId={selectedItem} model={model}
              onStoreChange={setStoreId} onItemChange={setSelectedItem} onModelChange={setModel}
              items={items} itemsLoading={itemsLoading}
            />
            <Button onClick={() => selectedItem && setExplainKey((k) => k + 1)} disabled={!selectedItem || exLoading}>
              {exLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Brain className="h-4 w-4" />}
              Explain Forecast
            </Button>
          </div>
        </CardContent>
      </Card>

      {exError && (
        <Card className="border-destructive">
          <CardContent className="py-4 text-center text-destructive text-sm">{exError.message}</CardContent>
        </Card>
      )}

      <AnimatePresence>
        {explain && (
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-4">
            {/* Prediction Info */}
            <div className="grid gap-4 md:grid-cols-3">
              <Card>
                <CardHeader className="pb-2"><CardDescription>Prediction</CardDescription>
                  <CardTitle className="text-3xl">{explain.explanation.prediction?.toFixed(2)}</CardTitle>
                </CardHeader>
                <CardContent><p className="text-sm text-muted-foreground">units/day</p></CardContent>
              </Card>
              <Card>
                <CardHeader className="pb-2"><CardDescription>Base Value</CardDescription>
                  <CardTitle className="text-3xl">{explain.explanation.base_value?.toFixed(2)}</CardTitle>
                </CardHeader>
                <CardContent><p className="text-sm text-muted-foreground">expected average</p></CardContent>
              </Card>
              <Card>
                <CardHeader className="pb-2"><CardDescription>Latest Actual Sales</CardDescription>
                  <CardTitle className="text-3xl">{explain.latest_actual_sales}</CardTitle>
                </CardHeader>
                <CardContent>
                  <Badge variant="outline">{model.toUpperCase()}</Badge>
                  <Badge variant="outline" className="ml-1">{explain.latency_ms}ms</Badge>
                </CardContent>
              </Card>
            </div>

            {/* SHAP Waterfall */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Brain className="h-5 w-5" /> SHAP Feature Contributions
                </CardTitle>
                <CardDescription>Top 5 positive and negative drivers of the prediction</CardDescription>
              </CardHeader>
              <CardContent>
                <ChartContainer config={{ impact: { label: "Impact" } }} className="h-[350px] w-full">
                  <BarChart data={chartData} layout="vertical" margin={{ left: 10, right: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                    <XAxis type="number" />
                    <YAxis dataKey="feature" type="category" width={150} tick={{ fontSize: 11 }} />
                    <ChartTooltip content={<ChartTooltipContent />} />
                    <ReferenceLine x={0} stroke="hsl(var(--foreground))" strokeWidth={1.5} />
                    <Bar dataKey="impact" radius={[0, 4, 4, 0]}>
                      {chartData.map((d, i) => (
                        <Cell key={i} fill={d.impact >= 0 ? "hsl(142, 76%, 36%)" : "hsl(0, 84%, 60%)"} />
                      ))}
                    </Bar>
                  </BarChart>
                </ChartContainer>
                <div className="flex gap-4 mt-2 text-xs text-muted-foreground">
                  <span className="flex items-center gap-1"><span className="h-3 w-3 rounded-sm bg-green-600" /> Positive (increases prediction)</span>
                  <span className="flex items-center gap-1"><span className="h-3 w-3 rounded-sm bg-red-500" /> Negative (decreases prediction)</span>
                </div>
              </CardContent>
            </Card>

            {/* Interpretation */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2"><Info className="h-5 w-5" /> Interpretation</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm leading-relaxed">{explain.interpretation}</p>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}