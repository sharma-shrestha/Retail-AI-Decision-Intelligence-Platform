"use client";

import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { STORE_IDS, MODEL_OPTIONS } from "@/lib/constants";

export function DataSelectors({
  storeId, itemId, model,
  onStoreChange, onItemChange, onModelChange,
  items, itemsLoading = false, showModel = true,
}) {
  return (
    <div className="flex flex-wrap items-end gap-3">
      <div className="flex flex-col gap-1.5">
        <label className="text-xs font-medium text-muted-foreground">Store</label>
        <Select value={storeId} onValueChange={onStoreChange}>
          <SelectTrigger className="w-[130px]">
            <SelectValue placeholder="Store" />
          </SelectTrigger>
          <SelectContent>
            {STORE_IDS.map((s) => (
              <SelectItem key={s} value={s}>{s}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className="flex flex-col gap-1.5">
        <label className="text-xs font-medium text-muted-foreground">Product</label>
        <Select value={itemId} onValueChange={onItemChange} disabled={itemsLoading}>
          <SelectTrigger className="w-[200px]">
            <SelectValue placeholder={itemsLoading ? "Loading..." : "Product"} />
          </SelectTrigger>
          <SelectContent>
            {items.map((item) => (
              <SelectItem key={item} value={item}>{item}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {showModel && (
        <div className="flex flex-col gap-1.5">
          <label className="text-xs font-medium text-muted-foreground">Model</label>
          <Select value={model} onValueChange={onModelChange}>
            <SelectTrigger className="w-[150px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {MODEL_OPTIONS.map((m) => (
                <SelectItem key={m.value} value={m.value}>{m.label}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      )}
    </div>
  );
}