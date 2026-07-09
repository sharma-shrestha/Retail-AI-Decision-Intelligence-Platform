"use client";

import { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchInventory, fetchTopProducts } from "@/lib/api";
import { DataSelectors } from "./data-selectors";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Package, Shield, AlertTriangle, ShoppingCart, Loader2,
  TrendingDown, CheckCircle2, ArrowRight,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

function StatusBadge({ status }) {
  const map = {
    healthy:    { label: "Healthy",    cls: "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400" },
    low:        { label: "Low Stock",  cls: "bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-400" },
    critical:   { label: "Critical",   cls: "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400" },
    out_of_stock: { label: "Out of Stock", cls: "bg-red-200 text-red-900 dark:bg-red-900/50 dark:text-red-300" },
  };
  const s = map[status] || map.healthy;
  return <span className={"inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium " + s.cls}>{s.label}</span>;
}

function UrgencyBadge({ urgency }) {
  const map = {
    low:      "secondary",
    medium:   "outline",
    high:     "outline",
    critical: "destructive",
    urgent:   "destructive",
  };
  return <Badge variant={map[urgency] || "secondary"}>{urgency}</Badge>;
}

export function InventoryTab() {
  const [storeId, setStoreId] = useState("CA_1");
  const [model, setModel] = useState("catboost");
  const [leadTime, setLeadTime] = useState("7");
  const [currentStock, setCurrentStock] = useState("0");
  const [invKey, setInvKey] = useState(0);
  const [selectedItem, setSelectedItem] = useState("");

  const { data: topData, isLoading: itemsLoading } = useQuery({
    queryKey: ["top-products-inv"],
    queryFn: () => fetchTopProducts(50, "total_sales"),
  });

  const items = useMemo(() => {
    if (!topData?.products) return [];
    return [...new Set(topData.products.map((p) => p.item_id))];
  }, [topData]);

  useMemo(() => {
    if (items.length > 0 && !selectedItem) setSelectedItem(items[0]);
  }, [items]);

  const { data: inv, isLoading: invLoading, error: invError } = useQuery({
    queryKey: ["inventory", storeId, selectedItem, model, leadTime, currentStock, invKey],
    queryFn: () => fetchInventory({
      store_id: storeId, item_id: selectedItem, model,
      current_stock: parseFloat(currentStock) || 0,
      lead_time_days: parseInt(leadTime) || 7,
    }),
    enabled: !!selectedItem && invKey > 0,
  });

  const metrics = inv ? [
    { icon: TrendingDown, label: "Daily Forecast", value: inv.daily_forecast?.toFixed(4), color: "" },
    { icon: Shield, label: "Safety Stock", value: inv.safety_stock?.toFixed(2), color: "" },
    { icon: ShoppingCart, label: "Reorder Point", value: inv.reorder_point?.toFixed(2), color: "" },
    { icon: Package, label: "Recommended Order Qty", value: inv.recommended_order_qty?.toFixed(2), color: "text-primary font-bold" },
  ] : [];

  return (
    <div className="space-y-6">
      {/* Controls */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex flex-wrap items-end gap-3">
            <DataSelectors
              storeId={storeId} itemId={selectedItem} model={model}
              onStoreChange={setStoreId} onItemChange={setSelectedItem} onModelChange={setModel}
              items={items} itemsLoading={itemsLoading}
            />
            <div className="flex flex-col gap-1.5">
              <label className="text-xs font-medium text-muted-foreground">Lead Time (days)</label>
              <Input type="number" value={leadTime} onChange={(e) => setLeadTime(e.target.value)} className="w-[100px] h-9" min="1" />
            </div>
            <div className="flex flex-col gap-1.5">
              <label className="text-xs font-medium text-muted-foreground">Current Stock</label>
              <Input type="number" value={currentStock} onChange={(e) => setCurrentStock(e.target.value)} className="w-[100px] h-9" min="0" />
            </div>
            <Button onClick={() => selectedItem && setInvKey((k) => k + 1)} disabled={!selectedItem || invLoading}>
              {invLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Package className="h-4 w-4" />}
              Get Recommendation
            </Button>
          </div>
        </CardContent>
      </Card>

      {invError && (
        <Card className="border-destructive">
          <CardContent className="py-4 text-center text-destructive text-sm">{invError.message}</CardContent>
        </Card>
      )}

      <AnimatePresence>
        {inv && (
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-4">
            {/* Status + Urgency */}
            <Card>
              <CardContent className="pt-6">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    {inv.stock_status === "healthy" || inv.stock_status === "low"
                      ? <CheckCircle2 className="h-8 w-8 text-green-600" />
                      : <AlertTriangle className="h-8 w-8 text-red-500" />}
                    <div>
                      <p className="font-semibold text-lg">{selectedItem} at {storeId}</p>
                      <p className="text-sm text-muted-foreground">
                        Daily demand: {inv.daily_demand?.toFixed(4)} units
                      </p>
                    </div>
                  </div>
                  <div className="text-right space-y-1">
                    <StatusBadge status={inv.stock_status} />
                    <div><UrgencyBadge urgency={inv.urgency} /></div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Metric Cards */}
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              {metrics.map(({ icon: I, label, value, color }) => (
                <Card key={label}>
                  <CardContent className="pt-6">
                    <div className="flex items-center gap-2 text-sm text-muted-foreground mb-1">
                      <I className="h-4 w-4" />{label}
                    </div>
                    <p className={"text-2xl font-bold " + color}>{value}</p>
                  </CardContent>
                </Card>
              ))}
            </div>

            {/* Reasoning */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2"><ArrowRight className="h-5 w-5" /> Recommendation Reasoning</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm leading-relaxed">{inv.reasoning}</p>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}