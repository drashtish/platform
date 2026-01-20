"""
Analytics module for UIDAI Dashboard.
Shared preprocessing logic for analytical consistency between notebook and dashboard.

EXACT MATCH with UIDAI_HACKATHON_FINAL_SUBMISSION.ipynb Section 2:
- Same CANONICAL_STATES (36 States/UTs)
- Same STATE_MAPPING and DISTRICT_MAPPING
- Same clean_dataset() function
- Same normalization functions
"""

from .preprocessing import (
    load_enrolment_data,
    load_biometric_data,
    load_demographic_data,
    build_risk_dataframe,
    clean_dataset,
    normalize_series,
    normalize_state,
    normalize_district,
    get_state_mapping,
    get_region,
    STATE_MAPPING,
    DISTRICT_MAPPING,
    CANONICAL_STATES,
    REGION_MAPPING,
    EPSILON
)

from .feature_engineering import (
    min_max_normalize,
    z_score_normalize,
    robust_normalize,
    compute_aesi_components,
    compute_aesi,
    compute_busi_components,
    compute_busi,
    classify_risk,
    compute_risk_capacity_quadrant,
    extract_temporal_features,
    compute_rolling_features,
    add_geographic_features,
    compute_ghost_center_features,
    compute_leaderboard_score,
    AESI_WEIGHTS,
    BUSI_WEIGHTS,
    RISK_THRESHOLDS
)

__all__ = [
    # Preprocessing
    'load_enrolment_data',
    'load_biometric_data', 
    'load_demographic_data',
    'build_risk_dataframe',
    'clean_dataset',
    'normalize_series',
    'normalize_state',
    'normalize_district',
    'get_state_mapping',
    'get_region',
    'STATE_MAPPING',
    'DISTRICT_MAPPING',
    'CANONICAL_STATES',
    'REGION_MAPPING',
    'EPSILON',
    
    # Feature Engineering
    'min_max_normalize',
    'z_score_normalize',
    'robust_normalize',
    'compute_aesi_components',
    'compute_aesi',
    'compute_busi_components',
    'compute_busi',
    'classify_risk',
    'compute_risk_capacity_quadrant',
    'extract_temporal_features',
    'compute_rolling_features',
    'add_geographic_features',
    'compute_ghost_center_features',
    'compute_leaderboard_score',
    'AESI_WEIGHTS',
    'BUSI_WEIGHTS',
    'RISK_THRESHOLDS'
]
