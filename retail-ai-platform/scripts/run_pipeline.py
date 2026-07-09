"""
End-to-End Pipeline Runner
============================
Run all modules in sequence: ETL → Features → Train → (optionally) start API.
"""

import gc
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_full_pipeline(start_api: bool = False):
    """Run the complete Phase 1 pipeline."""
    logger.info("=" * 60)
    logger.info("  RETAIL AI PLATFORM — PHASE 1 PIPELINE")
    logger.info("=" * 60)

    # Module 1: Data Engineering
    logger.info("\n[1/4] Running Data Engineering...")
    from src.data_engineering.etl import run_etl
    df = run_etl()
    del df
    gc.collect()

    # Module 2: Feature Engineering
    logger.info("\n[2/4] Running Feature Engineering...")
    from src.feature_engineering.features import run_feature_engineering
    train, val, features = run_feature_engineering()
    del train, val
    gc.collect()

    # Module 3: Model Training
    logger.info("\n[3/4] Running Model Training...")
    from src.models.train import run_training
    results = run_training()
    gc.collect()

    # Module 4: SHAP Explainability (quick test)
    logger.info("\n[4/4] Testing SHAP Explainability...")
    from src.explainability.shap_explainer import SHAPExplainer
    from src.models.train import load_data
    _, _, X_val, _, _ = load_data()
    explainer = SHAPExplainer(results["best"])
    explanation = explainer.explain_single(X_val, row_idx=0)
    logger.info(f"  Test prediction: {explanation['prediction']} (explained)")
    del explainer, X_val
    gc.collect()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("  PHASE 1 COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"  Models trained:  {', '.join(results['models'])}")
    logger.info(f"  Best model:      {results['best']}")
    for name, m in results["results"].items():
        logger.info(f"    {name}: RMSE={m['rmse']}, MAE={m['mae']}, RMSSE={m['rmsse']}")
    logger.info(f"  Features:        {len(features)}")
    logger.info("=" * 60)

    if start_api:
        logger.info("\nStarting FastAPI server...")
        import uvicorn
        from src.utils.config import API_HOST, API_PORT
        uvicorn.run("src.api.main:app", host=API_HOST, port=API_PORT, reload=False)


if __name__ == "__main__":
    start_api = "--api" in sys.argv or "-a" in sys.argv
    run_full_pipeline(start_api=start_api)