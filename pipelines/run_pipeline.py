"""
UIDAI Data Pipeline Runner
===========================
Converts raw CSV data into certified analytical datasets.
Ensures notebook-dashboard analytical consistency.
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
from datetime import datetime
import json
import hashlib
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics.preprocessing import (
    load_enrolment_data,
    load_biometric_data,
    load_demographic_data,
    build_risk_dataframe
)

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_BASE = BASE_DIR.parent  # Parent of uidai_dashboard
CERTIFIED_DIR = BASE_DIR / "data" / "certified"

OUTPUT_FILES = {
    "enrolment": CERTIFIED_DIR / "enrolment_clean.parquet",
    "biometric": CERTIFIED_DIR / "biometric_clean.parquet", 
    "demographic": CERTIFIED_DIR / "demographic_clean.parquet",
    "master": CERTIFIED_DIR / "master_dataset.parquet",
    "metadata": CERTIFIED_DIR / "pipeline_metadata.json"
}


def run_pipeline(force: bool = False):
    """Run the complete data certification pipeline."""
    print("=" * 60)
    print("UIDAI DATA CERTIFICATION PIPELINE")
    print("=" * 60)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Data source: {DATA_BASE}")
    print(f"Output: {CERTIFIED_DIR}")
    print()
    
    # Ensure output directory exists
    CERTIFIED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Stage 1: Load raw data using existing preprocessing
    print("[STAGE 1] Loading raw data...")
    
    # Bypass Streamlit cache for pipeline
    df_enrol = load_enrolment_data.__wrapped__(str(DATA_BASE), False)
    print(f"  Enrollment: {len(df_enrol):,} rows")
    
    df_bio = load_biometric_data.__wrapped__(str(DATA_BASE), False)
    print(f"  Biometric: {len(df_bio):,} rows")
    
    df_demo = load_demographic_data.__wrapped__(str(DATA_BASE), False)
    print(f"  Demographic: {len(df_demo):,} rows")
    
    # Stage 2: Build risk dataframe with indices
    print("\n[STAGE 2] Computing risk indices...")
    risk_df = build_risk_dataframe.__wrapped__(df_enrol, df_bio, df_demo)
    print(f"  Master dataset: {len(risk_df):,} districts")
    print(f"  States: {risk_df['state'].nunique()}")
    
    # Stage 3: Save certified parquet files
    print("\n[STAGE 3] Saving certified datasets...")
    
    if len(df_enrol) > 0:
        df_enrol.to_parquet(OUTPUT_FILES["enrolment"], index=False)
        print(f"  Saved: {OUTPUT_FILES['enrolment'].name}")
    
    if len(df_bio) > 0:
        df_bio.to_parquet(OUTPUT_FILES["biometric"], index=False)
        print(f"  Saved: {OUTPUT_FILES['biometric'].name}")
    
    if len(df_demo) > 0:
        df_demo.to_parquet(OUTPUT_FILES["demographic"], index=False)
        print(f"  Saved: {OUTPUT_FILES['demographic'].name}")
    
    if len(risk_df) > 0:
        risk_df.to_parquet(OUTPUT_FILES["master"], index=False)
        print(f"  Saved: {OUTPUT_FILES['master'].name}")
    
    # Stage 4: Generate metadata
    print("\n[STAGE 4] Generating metadata...")
    
    metadata = {
        "pipeline_version": "1.0.0",
        "run_timestamp": datetime.now().isoformat(),
        "data_source": str(DATA_BASE),
        "row_counts": {
            "enrolment": len(df_enrol),
            "biometric": len(df_bio),
            "demographic": len(df_demo),
            "master": len(risk_df)
        },
        "unique_states": int(risk_df["state"].nunique()) if len(risk_df) > 0 else 0,
        "unique_districts": len(risk_df),
        "index_summary": {}
    }
    
    # Add index statistics
    for idx in ["AESI", "BUSI", "ALSI", "DULI", "EPI", "Ghost_Score", "Leaderboard_Score"]:
        if idx in risk_df.columns:
            metadata["index_summary"][idx] = {
                "mean": float(risk_df[idx].mean()),
                "median": float(risk_df[idx].median()),
                "min": float(risk_df[idx].min()),
                "max": float(risk_df[idx].max()),
                "std": float(risk_df[idx].std())
            }
    
    # Risk distribution
    if "AESI" in risk_df.columns:
        metadata["risk_distribution"] = {
            "critical": int((risk_df["AESI"] >= 75).sum()),
            "high": int(((risk_df["AESI"] >= 50) & (risk_df["AESI"] < 75)).sum()),
            "moderate": int(((risk_df["AESI"] >= 25) & (risk_df["AESI"] < 50)).sum()),
            "low": int((risk_df["AESI"] < 25).sum())
        }
    
    with open(OUTPUT_FILES["metadata"], "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved: {OUTPUT_FILES['metadata'].name}")
    
    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Master dataset: {len(risk_df):,} districts across {metadata['unique_states']} states")
    if "risk_distribution" in metadata:
        rd = metadata["risk_distribution"]
        print(f"Risk distribution: Critical={rd['critical']}, High={rd['high']}, Moderate={rd['moderate']}, Low={rd['low']}")
    print(f"\nCertified data saved to: {CERTIFIED_DIR}")
    print("Dashboard should now read ONLY from certified/*.parquet")
    
    return metadata


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="UIDAI Data Pipeline")
    parser.add_argument("--force", action="store_true", help="Force run even if no changes")
    args = parser.parse_args()
    
    run_pipeline(force=args.force)
