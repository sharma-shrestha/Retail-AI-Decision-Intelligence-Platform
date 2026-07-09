"""
Module 5 — Inventory Intelligence Engine
==========================================
Converts demand forecasts into actionable inventory recommendations.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def compute_safety_stock(
    demand_std: float,
    lead_time_days: int = 7,
    service_level: float = 0.95,
) -> float:
    """Compute safety stock using the standard formula.

    Safety Stock = z × σ × √lead_time
    z = 1.645 for 95% service level
    """
    z_score = 1.645 if service_level >= 0.95 else 1.28  # 95% or 90%
    return round(z_score * demand_std * np.sqrt(lead_time_days), 2)


def generate_inventory_recommendation(
    forecasted_demand: float,
    current_stock: float | None,
    demand_std: float = 1.0,
    lead_time_days: int = 7,
    service_level: float = 0.95,
    days_ahead: int = 28,
) -> dict:
    """Generate an inventory action recommendation.

    Args:
        forecasted_demand: Predicted demand for the next period
        current_stock: Current inventory (optional)
        demand_std: Standard deviation of demand (for safety stock)
        lead_time_days: Replenishment lead time
        service_level: Desired service level (0.0-1.0)
        days_ahead: Forecast horizon in days

    Returns:
        Dictionary with order recommendation and reasoning
    """
    daily_demand = forecasted_demand / days_ahead
    safety_stock = compute_safety_stock(demand_std, lead_time_days, service_level)
    reorder_point = (daily_demand * lead_time_days) + safety_stock
    recommended_order = max(0, round(forecasted_demand + safety_stock - (current_stock or 0), 0))

    # Determine action
    if current_stock is not None:
        stock_status = "adequate" if current_stock >= reorder_point else "low"
        urgency = "normal"
        if current_stock < safety_stock:
            stock_status = "critical"
            urgency = "urgent"
        elif current_stock < reorder_point:
            urgency = "moderate"
    else:
        stock_status = "unknown"
        urgency = "info"

    return {
        "forecasted_demand": round(float(forecasted_demand), 2),
        "daily_demand": round(float(daily_demand), 2),
        "safety_stock": safety_stock,
        "reorder_point": round(reorder_point, 2),
        "recommended_order_qty": recommended_order,
        "current_stock": float(current_stock) if current_stock is not None else None,
        "stock_status": stock_status,
        "urgency": urgency,
        "lead_time_days": lead_time_days,
        "service_level": service_level,
        "reasoning": (
            f"Expected demand of {forecasted_demand:.0f} units over {days_ahead} days "
            f"(~{daily_demand:.1f}/day). "
            f"Safety stock of {safety_stock:.0f} units accounts for demand variability "
            f"(σ={demand_std:.2f}) at {service_level:.0%} service level with {lead_time_days}-day lead time. "
            + (
                f"Current stock ({current_stock:.0f}) is {'below reorder point, '
                f'replenishment recommended' if stock_status != 'adequate' else 'sufficient'}"
                if current_stock is not None
                else "No current stock data provided."
            )
        ),
    }


def batch_inventory_recommendations(
    forecasts_df: pd.DataFrame,
    demand_stds: pd.Series | None = None,
    lead_time_days: int = 7,
) -> list[dict]:
    """Generate inventory recommendations for multiple products.

    Args:
        forecasts_df: DataFrame with columns [store_id, item_id, predicted_sales, ...]
        demand_stds: Optional Series of demand std per product-store
        lead_time_days: Replenishment lead time

    Returns:
        List of recommendation dicts
    """
    results = []

    for _, row in forecasts_df.iterrows():
        std = float(demand_stds.iloc[_]) if demand_stds is not None else 1.0
        rec = generate_inventory_recommendation(
            forecasted_demand=float(row.get("predicted_sales", row.get("prediction", 0))),
            current_stock=row.get("current_stock", None),
            demand_std=std,
            lead_time_days=lead_time_days,
        )
        rec["store_id"] = row["store_id"]
        rec["item_id"] = row["item_id"]
        results.append(rec)

    return results


if __name__ == "__main__":
    # Demo
    rec = generate_inventory_recommendation(
        forecasted_demand=250,
        current_stock=180,
        demand_std=5.0,
    )
    for k, v in rec.items():
        print(f"  {k}: {v}")