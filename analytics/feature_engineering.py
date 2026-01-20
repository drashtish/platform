"""
================================================================================
UIDAI GOVERNANCE INTELLIGENCE PLATFORM - FEATURE ENGINEERING MODULE
================================================================================
Advanced feature engineering for governance analytics:
- AESI (Aadhaar Ecosystem Stress Index) components
- BUSI (Biometric Update Stress Index) components
- ALSI (Aadhaar Lifecycle Stress Index) components
- Risk scoring and classification
- Temporal feature extraction
- Geographic feature aggregation

SHARED MODULE: Used by both notebook and dashboard for analytical consistency.
================================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta


# =============================================================================
# CONSTANTS
# =============================================================================

EPSILON = 1e-10  # Small value to avoid division by zero

# AESI Component Weights (based on domain expertise)
AESI_WEIGHTS = {
    'enrollment_volume': 0.25,
    'center_density': 0.20,
    'rejection_rate': 0.20,
    'processing_time': 0.15,
    'capacity_utilization': 0.20
}

# BUSI Component Weights
BUSI_WEIGHTS = {
    'update_volume': 0.25,
    'biometric_failure_rate': 0.30,
    'processing_backlog': 0.25,
    'equipment_age_factor': 0.20
}

# Risk Classification Thresholds
RISK_THRESHOLDS = {
    'critical': 0.8,
    'high': 0.6,
    'medium': 0.4,
    'low': 0.2
}

# Region Mapping for Geographic Features
REGION_MAPPING = {
    'North': ['Jammu and Kashmir', 'Ladakh', 'Himachal Pradesh', 'Punjab', 
              'Chandigarh', 'Uttarakhand', 'Haryana', 'NCT of Delhi', 'Uttar Pradesh'],
    'South': ['Andhra Pradesh', 'Karnataka', 'Kerala', 'Tamil Nadu', 
              'Telangana', 'Puducherry', 'Lakshadweep'],
    'East': ['Bihar', 'Jharkhand', 'Odisha', 'West Bengal', 
             'Andaman and Nicobar Islands'],
    'West': ['Rajasthan', 'Gujarat', 'Goa', 'Maharashtra', 
             'Dadra and Nagar Haveli and Daman and Diu'],
    'Central': ['Madhya Pradesh', 'Chhattisgarh'],
    'Northeast': ['Assam', 'Sikkim', 'Nagaland', 'Meghalaya', 'Manipur', 
                  'Mizoram', 'Tripura', 'Arunachal Pradesh']
}


# =============================================================================
# NORMALIZATION FUNCTIONS
# =============================================================================

def min_max_normalize(series: pd.Series, inverse: bool = False) -> pd.Series:
    """
    Apply min-max normalization to a series.
    
    Args:
        series: Input pandas Series
        inverse: If True, high values become low scores (for negative metrics)
    
    Returns:
        Normalized series with values between 0 and 1
    """
    min_val = series.min()
    max_val = series.max()
    
    if max_val - min_val < EPSILON:
        return pd.Series(0.5, index=series.index)
    
    normalized = (series - min_val) / (max_val - min_val + EPSILON)
    
    if inverse:
        normalized = 1 - normalized
    
    return normalized


def z_score_normalize(series: pd.Series) -> pd.Series:
    """
    Apply z-score normalization to a series.
    
    Args:
        series: Input pandas Series
    
    Returns:
        Z-score normalized series
    """
    mean_val = series.mean()
    std_val = series.std()
    
    if std_val < EPSILON:
        return pd.Series(0, index=series.index)
    
    return (series - mean_val) / (std_val + EPSILON)


def robust_normalize(series: pd.Series) -> pd.Series:
    """
    Apply robust normalization using median and IQR.
    More resistant to outliers than min-max or z-score.
    
    Args:
        series: Input pandas Series
    
    Returns:
        Robustly normalized series
    """
    median_val = series.median()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    
    if iqr < EPSILON:
        return pd.Series(0, index=series.index)
    
    return (series - median_val) / (iqr + EPSILON)


# =============================================================================
# AESI (AADHAAR ECOSYSTEM STRESS INDEX) FEATURES
# =============================================================================

def compute_aesi_components(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute individual AESI components for each district/state.
    
    Expected columns:
        - total_enrolment: Total enrollments
        - centre_count: Number of enrollment centers
        - rejection_count: Number of rejected applications
        - avg_processing_days: Average processing time
        - population: Population estimate
    
    Returns:
        DataFrame with AESI component scores
    """
    result = pd.DataFrame(index=df.index)
    
    # 1. Enrollment Volume Score (higher = more stress)
    if 'total_enrolment' in df.columns:
        result['enrollment_volume_score'] = min_max_normalize(df['total_enrolment'])
    
    # 2. Center Density Score (lower density = more stress)
    if 'centre_count' in df.columns and 'population' in df.columns:
        centers_per_lakh = (df['centre_count'] / (df['population'] / 100000 + EPSILON))
        result['center_density_score'] = min_max_normalize(centers_per_lakh, inverse=True)
    
    # 3. Rejection Rate Score (higher rejection = more stress)
    if 'rejection_count' in df.columns and 'total_enrolment' in df.columns:
        rejection_rate = df['rejection_count'] / (df['total_enrolment'] + EPSILON)
        result['rejection_rate_score'] = min_max_normalize(rejection_rate)
    
    # 4. Processing Time Score (longer time = more stress)
    if 'avg_processing_days' in df.columns:
        result['processing_time_score'] = min_max_normalize(df['avg_processing_days'])
    
    # 5. Capacity Utilization Score (over-utilized = more stress)
    if 'total_enrolment' in df.columns and 'centre_count' in df.columns:
        enrolments_per_center = df['total_enrolment'] / (df['centre_count'] + EPSILON)
        result['capacity_utilization_score'] = min_max_normalize(enrolments_per_center)
    
    return result


def compute_aesi(df: pd.DataFrame, weights: Optional[Dict[str, float]] = None) -> pd.Series:
    """
    Compute the composite AESI score.
    
    Args:
        df: DataFrame with required columns
        weights: Optional custom weights for components
    
    Returns:
        Series with AESI scores (0-100 scale)
    """
    if weights is None:
        weights = AESI_WEIGHTS
    
    components = compute_aesi_components(df)
    
    aesi = pd.Series(0, index=df.index, dtype=float)
    total_weight = 0
    
    for component, weight in weights.items():
        col_name = f"{component}_score"
        if col_name in components.columns:
            aesi += components[col_name] * weight
            total_weight += weight
    
    if total_weight > 0:
        aesi = (aesi / total_weight) * 100
    
    return aesi.round(2)


# =============================================================================
# BUSI (BIOMETRIC UPDATE STRESS INDEX) FEATURES
# =============================================================================

def compute_busi_components(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute individual BUSI components.
    
    Expected columns:
        - biometric_updates: Total biometric update requests
        - fingerprint_failures: Fingerprint capture failures
        - iris_failures: Iris capture failures
        - pending_updates: Backlog of pending updates
        - equipment_age_years: Average age of biometric equipment
    
    Returns:
        DataFrame with BUSI component scores
    """
    result = pd.DataFrame(index=df.index)
    
    # 1. Update Volume Score
    if 'biometric_updates' in df.columns:
        result['update_volume_score'] = min_max_normalize(df['biometric_updates'])
    
    # 2. Biometric Failure Rate
    if all(col in df.columns for col in ['fingerprint_failures', 'iris_failures', 'biometric_updates']):
        total_failures = df['fingerprint_failures'] + df['iris_failures']
        failure_rate = total_failures / (df['biometric_updates'] + EPSILON)
        result['biometric_failure_score'] = min_max_normalize(failure_rate)
    
    # 3. Processing Backlog Score
    if 'pending_updates' in df.columns and 'biometric_updates' in df.columns:
        backlog_ratio = df['pending_updates'] / (df['biometric_updates'] + EPSILON)
        result['backlog_score'] = min_max_normalize(backlog_ratio)
    
    # 4. Equipment Age Factor
    if 'equipment_age_years' in df.columns:
        result['equipment_age_score'] = min_max_normalize(df['equipment_age_years'])
    
    return result


def compute_busi(df: pd.DataFrame, weights: Optional[Dict[str, float]] = None) -> pd.Series:
    """
    Compute the composite BUSI score.
    
    Args:
        df: DataFrame with required columns
        weights: Optional custom weights for components
    
    Returns:
        Series with BUSI scores (0-100 scale)
    """
    if weights is None:
        weights = BUSI_WEIGHTS
    
    components = compute_busi_components(df)
    
    busi = pd.Series(0, index=df.index, dtype=float)
    total_weight = 0
    
    weight_mapping = {
        'update_volume': 'update_volume_score',
        'biometric_failure_rate': 'biometric_failure_score',
        'processing_backlog': 'backlog_score',
        'equipment_age_factor': 'equipment_age_score'
    }
    
    for component, weight in weights.items():
        col_name = weight_mapping.get(component)
        if col_name and col_name in components.columns:
            busi += components[col_name] * weight
            total_weight += weight
    
    if total_weight > 0:
        busi = (busi / total_weight) * 100
    
    return busi.round(2)


# =============================================================================
# RISK CLASSIFICATION
# =============================================================================

def classify_risk(score: Union[float, pd.Series], 
                  thresholds: Optional[Dict[str, float]] = None) -> Union[str, pd.Series]:
    """
    Classify risk level based on score.
    
    Args:
        score: Numeric score (0-100) or Series of scores
        thresholds: Optional custom thresholds
    
    Returns:
        Risk classification string or Series
    """
    if thresholds is None:
        thresholds = RISK_THRESHOLDS
    
    # Scale thresholds to 0-100
    scaled_thresholds = {k: v * 100 for k, v in thresholds.items()}
    
    if isinstance(score, pd.Series):
        conditions = [
            score >= scaled_thresholds['critical'],
            score >= scaled_thresholds['high'],
            score >= scaled_thresholds['medium'],
            score >= scaled_thresholds['low']
        ]
        choices = ['🔴 Critical', '🟠 High', '🟡 Medium', '🟢 Low']
        return pd.Series(np.select(conditions, choices, default='⚪ Minimal'), index=score.index)
    else:
        if score >= scaled_thresholds['critical']:
            return '🔴 Critical'
        elif score >= scaled_thresholds['high']:
            return '🟠 High'
        elif score >= scaled_thresholds['medium']:
            return '🟡 Medium'
        elif score >= scaled_thresholds['low']:
            return '🟢 Low'
        else:
            return '⚪ Minimal'


def compute_risk_capacity_quadrant(df: pd.DataFrame, 
                                    risk_col: str = 'aesi', 
                                    capacity_col: str = 'centre_count') -> pd.Series:
    """
    Classify districts into risk-capacity quadrants.
    
    Quadrants:
        - Q1 (High Risk, Low Capacity): Priority intervention needed
        - Q2 (High Risk, High Capacity): Monitor closely
        - Q3 (Low Risk, Low Capacity): Capacity building opportunity
        - Q4 (Low Risk, High Capacity): Well-performing
    
    Returns:
        Series with quadrant classifications
    """
    risk_median = df[risk_col].median()
    capacity_median = df[capacity_col].median()
    
    conditions = [
        (df[risk_col] >= risk_median) & (df[capacity_col] < capacity_median),
        (df[risk_col] >= risk_median) & (df[capacity_col] >= capacity_median),
        (df[risk_col] < risk_median) & (df[capacity_col] < capacity_median),
        (df[risk_col] < risk_median) & (df[capacity_col] >= capacity_median)
    ]
    
    choices = [
        'Q1: High Risk, Low Capacity',
        'Q2: High Risk, High Capacity',
        'Q3: Low Risk, Low Capacity',
        'Q4: Low Risk, High Capacity'
    ]
    
    return pd.Series(np.select(conditions, choices, default='Unclassified'), index=df.index)


# =============================================================================
# TEMPORAL FEATURES
# =============================================================================

def extract_temporal_features(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """
    Extract temporal features from date column.
    
    Returns:
        DataFrame with temporal features added
    """
    result = df.copy()
    
    if date_col not in df.columns:
        return result
    
    # Ensure datetime type
    result[date_col] = pd.to_datetime(result[date_col])
    
    # Extract features
    result['year'] = result[date_col].dt.year
    result['month'] = result[date_col].dt.month
    result['quarter'] = result[date_col].dt.quarter
    result['day_of_week'] = result[date_col].dt.dayofweek
    result['is_weekend'] = result['day_of_week'].isin([5, 6]).astype(int)
    result['is_month_start'] = result[date_col].dt.is_month_start.astype(int)
    result['is_month_end'] = result[date_col].dt.is_month_end.astype(int)
    result['week_of_year'] = result[date_col].dt.isocalendar().week
    
    return result


def compute_rolling_features(df: pd.DataFrame, 
                              value_col: str, 
                              windows: List[int] = [7, 14, 30]) -> pd.DataFrame:
    """
    Compute rolling statistics for time series analysis.
    
    Args:
        df: DataFrame sorted by date
        value_col: Column to compute rolling features for
        windows: List of window sizes
    
    Returns:
        DataFrame with rolling features added
    """
    result = df.copy()
    
    for window in windows:
        result[f'{value_col}_rolling_mean_{window}d'] = result[value_col].rolling(window).mean()
        result[f'{value_col}_rolling_std_{window}d'] = result[value_col].rolling(window).std()
        result[f'{value_col}_rolling_min_{window}d'] = result[value_col].rolling(window).min()
        result[f'{value_col}_rolling_max_{window}d'] = result[value_col].rolling(window).max()
    
    return result


# =============================================================================
# GEOGRAPHIC FEATURES
# =============================================================================

def get_region(state: str) -> str:
    """
    Get the region for a given state.
    
    Args:
        state: State name
    
    Returns:
        Region name
    """
    for region, states in REGION_MAPPING.items():
        if state in states:
            return region
    return 'Other'


def add_geographic_features(df: pd.DataFrame, state_col: str = 'state') -> pd.DataFrame:
    """
    Add geographic features based on state.
    
    Returns:
        DataFrame with region column added
    """
    result = df.copy()
    
    if state_col in df.columns:
        result['region'] = result[state_col].apply(get_region)
    
    return result


# =============================================================================
# GHOST CENTER DETECTION FEATURES
# =============================================================================

def compute_ghost_center_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute features for ghost center detection.
    
    Ghost centers are enrollment centers with suspicious patterns:
    - Very high enrollment numbers with few rejections
    - Unusual temporal patterns
    - Geographic isolation
    
    Returns:
        DataFrame with ghost detection features
    """
    result = df.copy()
    
    # 1. Rejection Ratio Anomaly (too low is suspicious)
    if 'rejection_count' in df.columns and 'total_enrolment' in df.columns:
        rejection_ratio = df['rejection_count'] / (df['total_enrolment'] + EPSILON)
        result['rejection_ratio_zscore'] = z_score_normalize(rejection_ratio)
        result['low_rejection_flag'] = (result['rejection_ratio_zscore'] < -2).astype(int)
    
    # 2. Volume Anomaly (unusually high volume)
    if 'total_enrolment' in df.columns:
        result['volume_zscore'] = z_score_normalize(df['total_enrolment'])
        result['high_volume_flag'] = (result['volume_zscore'] > 2).astype(int)
    
    # 3. Combined Ghost Score
    if 'low_rejection_flag' in result.columns and 'high_volume_flag' in result.columns:
        result['ghost_score'] = (result['low_rejection_flag'] + result['high_volume_flag']) / 2 * 100
    
    return result


# =============================================================================
# LEADERBOARD SCORING
# =============================================================================

def compute_leaderboard_score(df: pd.DataFrame,
                               metrics: Dict[str, Tuple[str, bool]]) -> pd.Series:
    """
    Compute composite leaderboard score from multiple metrics.
    
    Args:
        df: DataFrame with metric columns
        metrics: Dict of {column_name: (weight, higher_is_better)}
    
    Returns:
        Series with leaderboard scores (0-100)
    """
    total_score = pd.Series(0, index=df.index, dtype=float)
    total_weight = 0
    
    for col, (weight, higher_is_better) in metrics.items():
        if col in df.columns:
            normalized = min_max_normalize(df[col], inverse=not higher_is_better)
            total_score += normalized * weight
            total_weight += weight
    
    if total_weight > 0:
        total_score = (total_score / total_weight) * 100
    
    return total_score.round(2)


# =============================================================================
# EXPORT ALL FUNCTIONS
# =============================================================================

__all__ = [
    # Normalization
    'min_max_normalize',
    'z_score_normalize',
    'robust_normalize',
    
    # AESI
    'compute_aesi_components',
    'compute_aesi',
    'AESI_WEIGHTS',
    
    # BUSI
    'compute_busi_components',
    'compute_busi',
    'BUSI_WEIGHTS',
    
    # Risk Classification
    'classify_risk',
    'compute_risk_capacity_quadrant',
    'RISK_THRESHOLDS',
    
    # Temporal Features
    'extract_temporal_features',
    'compute_rolling_features',
    
    # Geographic Features
    'get_region',
    'add_geographic_features',
    'REGION_MAPPING',
    
    # Ghost Detection
    'compute_ghost_center_features',
    
    # Leaderboard
    'compute_leaderboard_score',
    
    # Constants
    'EPSILON'
]
