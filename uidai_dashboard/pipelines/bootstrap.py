"""
================================================================================
UIDAI GOVERNANCE PLATFORM - DATA BOOTSTRAP MODULE
================================================================================
Self-initializing data layer for cloud deployment.

PROBLEM: Streamlit Cloud runs fresh containers without pre-built data files.
SOLUTION: Auto-detect missing certified data and rebuild on-demand.

This ensures:
- First user triggers automatic data build (~10-15 sec)
- Subsequent users load instantly from cache
- No "Oh no" crashes from missing files
================================================================================
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Paths
BASE_DIR = Path(__file__).parent.parent
CERTIFIED_DIR = BASE_DIR / "data" / "certified"

# Required certified files for dashboard operation
REQUIRED_FILES = [
    "master_dataset.parquet",
    "enrolment_clean.parquet",
    "biometric_clean.parquet",
    "demographic_clean.parquet",
]


def check_certified_data_exists() -> dict:
    """
    Check if all required certified data files exist.
    
    Returns:
        dict with 'exists' (bool) and 'missing' (list of missing files)
    """
    CERTIFIED_DIR.mkdir(parents=True, exist_ok=True)
    
    missing = []
    for filename in REQUIRED_FILES:
        filepath = CERTIFIED_DIR / filename
        if not filepath.exists():
            missing.append(filename)
        elif filepath.stat().st_size == 0:
            # File exists but is empty - treat as missing
            missing.append(filename)
    
    return {
        "exists": len(missing) == 0,
        "missing": missing,
        "certified_dir": str(CERTIFIED_DIR)
    }


def ensure_certified_data(verbose: bool = True) -> bool:
    """
    Ensure certified data exists. Rebuild if missing.
    
    This is the main entry point for data bootstrap.
    Call this at app startup to guarantee data availability.
    
    Args:
        verbose: Print status messages
    
    Returns:
        True if data is ready, False if rebuild failed
    """
    status = check_certified_data_exists()
    
    if status["exists"]:
        if verbose:
            print("✓ Certified data verified")
        return True
    
    if verbose:
        print(f"⚠ Missing certified data: {status['missing']}")
        print("Attempting to rebuild from pipeline...")
    
    try:
        # Import here to avoid circular imports
        from pipelines.run_pipeline import run_pipeline
        run_pipeline(force=True)
        
        # Verify rebuild succeeded
        status = check_certified_data_exists()
        if status["exists"]:
            if verbose:
                print("✓ Certified data rebuilt successfully")
            return True
        else:
            if verbose:
                print(f"✗ Rebuild incomplete. Still missing: {status['missing']}")
            return False
            
    except Exception as e:
        if verbose:
            print(f"✗ Pipeline rebuild failed: {str(e)}")
        return False


def get_data_status() -> dict:
    """
    Get detailed status of certified data for debugging.
    
    Returns:
        dict with file status, sizes, and timestamps
    """
    status = check_certified_data_exists()
    
    files_info = {}
    for filename in REQUIRED_FILES:
        filepath = CERTIFIED_DIR / filename
        if filepath.exists():
            stat = filepath.stat()
            files_info[filename] = {
                "exists": True,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified": stat.st_mtime
            }
        else:
            files_info[filename] = {
                "exists": False,
                "size_mb": 0,
                "modified": None
            }
    
    status["files"] = files_info
    return status


if __name__ == "__main__":
    # CLI usage: python -m pipelines.bootstrap
    print("=" * 60)
    print("UIDAI DATA BOOTSTRAP CHECK")
    print("=" * 60)
    
    status = get_data_status()
    print(f"\nCertified directory: {status['certified_dir']}")
    print(f"Data ready: {status['exists']}")
    
    if status['missing']:
        print(f"Missing files: {status['missing']}")
    
    print("\nFile details:")
    for filename, info in status['files'].items():
        if info['exists']:
            print(f"  ✓ {filename}: {info['size_mb']} MB")
        else:
            print(f"  ✗ {filename}: MISSING")
    
    if not status['exists']:
        print("\nRun with --rebuild to attempt data rebuild")
        if len(sys.argv) > 1 and sys.argv[1] == "--rebuild":
            ensure_certified_data()
