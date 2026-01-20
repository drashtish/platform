"""
================================================================================
UIDAI GOVERNANCE INTELLIGENCE DASHBOARD - RISK ENGINE
================================================================================
Purpose: Calculate governance risk indices and anomaly scores
Extracted from: UIDAI_HACKATHON_FINAL_SUBMISSION.ipynb

Key Metrics:
- Enrolment Pressure Index (EPI)
- Demographic Update Load Index (DULI)  
- Biometric Risk Score (BRS)
- Composite Aadhaar Risk Index (CARI)
- Aadhaar Enrollment Stress Index (AESI)
- Aadhaar Lifecycle Stress Index (ALSI)
================================================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

EPSILON = 1e-6


# =============================================================================
# ENROLMENT PRESSURE INDEX (EPI)
# =============================================================================

def calculate_epi(df_unified: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Enrolment Pressure Index.
    
    EPI measures the relative enrollment pressure on a district compared to
    national averages. High EPI indicates potential overload zones.
    
    Formula: EPI = (District_Enrollment / National_Avg) × Activity_Density_Factor
    """
    df = df_unified.copy()
    
    if 'total_enrollment' not in df.columns:
        df['total_enrollment'] = 0
    
    # National average enrollment per district
    national_avg = df['total_enrollment'].mean()
    
    # Calculate base EPI
    df['epi_base'] = df['total_enrollment'] / (national_avg + EPSILON)
    
    # Activity density factor (enrollments per pincode if available)
    if 'enroll_pincodes' in df.columns:
        df['activity_density'] = df['total_enrollment'] / (df['enroll_pincodes'].replace(0, 1))
        density_factor = df['activity_density'] / (df['activity_density'].mean() + EPSILON)
    else:
        density_factor = 1
    
    # Final EPI
    df['EPI'] = df['epi_base'] * density_factor
    
    # Normalize to 0-100 scale
    scaler = MinMaxScaler(feature_range=(0, 100))
    df['EPI_normalized'] = scaler.fit_transform(df[['EPI']].fillna(0))
    
    # Classify EPI
    df['EPI_category'] = pd.cut(
        df['EPI_normalized'],
        bins=[0, 25, 50, 75, 90, 100],
        labels=['Low', 'Moderate', 'Elevated', 'High', 'Critical'],
        include_lowest=True
    )
    
    return df


# =============================================================================
# DEMOGRAPHIC UPDATE LOAD INDEX (DULI)
# =============================================================================

def calculate_duli(df_unified: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Demographic Update Load Index.
    
    DULI measures the demographic update burden on a district.
    High DULI indicates districts with heavy address/name update volumes.
    
    Formula: DULI = (Total_Updates / Enrollment_Base) × Variance_Factor
    """
    df = df_unified.copy()
    
    if 'total_demo' not in df.columns:
        df['total_demo'] = 0
    
    # Calculate update intensity relative to enrollment base
    enrollment_base = df.get('total_enrollment', pd.Series([1] * len(df))).replace(0, 1)
    df['update_intensity'] = df['total_demo'] / enrollment_base
    
    # Calculate variance factor (higher variance = more stress)
    if 'demo_child' in df.columns and 'demo_adult' in df.columns:
        df['update_variance'] = df[['demo_child', 'demo_adult']].std(axis=1)
        variance_factor = 1 + (df['update_variance'] / (df['update_variance'].max() + EPSILON))
    else:
        variance_factor = 1
    
    # Final DULI
    df['DULI'] = df['update_intensity'] * variance_factor
    
    # Normalize to 0-100 scale
    scaler = MinMaxScaler(feature_range=(0, 100))
    df['DULI_normalized'] = scaler.fit_transform(df[['DULI']].fillna(0).replace([np.inf, -np.inf], 0))
    
    # Classify DULI
    df['DULI_category'] = pd.cut(
        df['DULI_normalized'],
        bins=[0, 25, 50, 75, 90, 100],
        labels=['Low', 'Moderate', 'Elevated', 'High', 'Critical'],
        include_lowest=True
    )
    
    return df


# =============================================================================
# BIOMETRIC RISK SCORE (BRS)
# =============================================================================

def calculate_brs(df_unified: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Biometric Risk Score.
    
    BRS assesses biometric verification compliance and potential fraud risk.
    
    Formula: BRS = (1 - Security_Ratio) × Child_Ratio_Factor
    
    Security Ratio = Biometric_Updates / Demographic_Updates
    Low security ratio indicates potential operators bypassing biometric checks.
    """
    df = df_unified.copy()
    
    # Security Ratio (biometric compliance)
    total_demo = df.get('total_demo', pd.Series([1] * len(df))).replace(0, 1)
    total_bio = df.get('total_bio', pd.Series([0] * len(df)))
    
    df['security_ratio'] = total_bio / (total_demo + EPSILON)
    df['security_ratio_capped'] = df['security_ratio'].clip(0, 2)
    
    # Child ratio factor (higher child biometrics = more MBU compliance needed)
    if 'bio_child' in df.columns and 'bio_adult' in df.columns:
        bio_total = df.get('total_bio', pd.Series([1] * len(df))).replace(0, 1)
        df['bio_child_ratio'] = df.get('bio_child', 0) / bio_total
        child_factor = 1 + df['bio_child_ratio']
    else:
        child_factor = 1
    
    # BRS calculation (higher = more risk)
    # Invert security ratio so low compliance = high risk
    df['BRS'] = (2 - df['security_ratio_capped']) * child_factor
    
    # Normalize to 0-100 scale
    scaler = MinMaxScaler(feature_range=(0, 100))
    df['BRS_normalized'] = scaler.fit_transform(df[['BRS']].fillna(0).replace([np.inf, -np.inf], 0))
    
    # Classify BRS (>0.75 = Critical, 0.5-0.75 = High, <0.5 = Normal)
    df['BRS_category'] = pd.cut(
        df['BRS_normalized'],
        bins=[0, 50, 75, 100],
        labels=['Normal', 'High', 'Critical'],
        include_lowest=True
    )
    
    return df


# =============================================================================
# COMPOSITE AADHAAR RISK INDEX (CARI)
# =============================================================================

def calculate_cari(df_unified: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Composite Aadhaar Risk Index.
    
    CARI = 0.4 × EPI + 0.3 × DULI + 0.3 × BRS
    
    This is the master risk score combining all three indices.
    """
    df = df_unified.copy()
    
    # Ensure component scores exist
    if 'EPI_normalized' not in df.columns:
        df = calculate_epi(df)
    if 'DULI_normalized' not in df.columns:
        df = calculate_duli(df)
    if 'BRS_normalized' not in df.columns:
        df = calculate_brs(df)
    
    # Calculate CARI with weighted average
    df['CARI'] = (
        0.4 * df['EPI_normalized'].fillna(0) +
        0.3 * df['DULI_normalized'].fillna(0) +
        0.3 * df['BRS_normalized'].fillna(0)
    )
    
    # Classify CARI
    df['CARI_category'] = pd.cut(
        df['CARI'],
        bins=[0, 25, 50, 75, 100],
        labels=['Low Risk', 'Moderate Risk', 'High Risk', 'Critical Risk'],
        include_lowest=True
    )
    
    return df


# =============================================================================
# AADHAAR ENROLLMENT STRESS INDEX (AESI)
# =============================================================================

def calculate_aesi(df_unified: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Aadhaar Enrollment Stress Index.
    
    AESI combines multiple risk dimensions:
    - Security compliance (25%)
    - Migration intensity (25%)
    - Service load (25%)
    - Coverage gap (25%)
    
    Higher AESI = More stress/risk
    """
    df = df_unified.copy()
    scaler = MinMaxScaler()
    
    # Component 1: Security Score (low security = high stress)
    total_demo = df.get('total_demo', pd.Series([1] * len(df))).replace(0, 1)
    total_bio = df.get('total_bio', pd.Series([0] * len(df)))
    security_ratio = total_bio / (total_demo + EPSILON)
    security_score = security_ratio.clip(0, 2)
    df['security_normalized'] = 1 - scaler.fit_transform(security_score.values.reshape(-1, 1)).flatten()
    
    # Component 2: Migration Stress (high migration intensity = high stress)
    enrollment_base = df.get('total_enrollment', pd.Series([1] * len(df))).replace(0, 1)
    df['migration_intensity'] = df.get('total_demo', 0) / (enrollment_base + EPSILON)
    migration_capped = df['migration_intensity'].clip(0, 20)
    df['migration_normalized'] = scaler.fit_transform(migration_capped.values.reshape(-1, 1)).flatten()
    
    # Component 3: Service Load Stress
    total_activity = df.get('total_activity', pd.Series([0] * len(df)))
    enroll_pincodes = df.get('enroll_pincodes', pd.Series([1] * len(df))).replace(0, 1)
    activity_per_pincode = total_activity / enroll_pincodes
    load_capped = activity_per_pincode.clip(0, activity_per_pincode.quantile(0.99) if len(activity_per_pincode) > 0 else 1)
    df['load_normalized'] = scaler.fit_transform(load_capped.values.reshape(-1, 1)).flatten()
    
    # Component 4: Coverage Gap
    data_completeness = df.get('data_completeness', pd.Series([3] * len(df)))
    df['coverage_normalized'] = 1 - (data_completeness / 3)
    
    # Calculate AESI - use .values to get numpy arrays
    df['AESI'] = (
        0.25 * df['security_normalized'].values +
        0.25 * df['migration_normalized'].values +
        0.25 * df['load_normalized'].values +
        0.25 * df['coverage_normalized'].values
    ) * 100
    
    # Classify AESI
    df['AESI_category'] = pd.cut(
        df['AESI'],
        bins=[0, 25, 50, 75, 100],
        labels=['Low Risk', 'Moderate Risk', 'High Risk', 'Critical Risk'],
        include_lowest=True
    )
    
    return df


# =============================================================================
# AADHAAR LIFECYCLE STRESS INDEX (ALSI)
# =============================================================================

def calculate_alsi(df_enroll: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Aadhaar Lifecycle Stress Index.
    
    ALSI predicts future MBU crisis by analyzing child enrollment patterns.
    Children enrolled today will need mandatory biometric updates in 5-7 years.
    
    Formula: ALSI = (Child_Enrollment × Days_Since_Start) / (Adult_Activity + 1)
    """
    if df_enroll is None or len(df_enroll) == 0:
        return pd.DataFrame()
    
    alsi_data = df_enroll.groupby(['state', 'district']).agg({
        'child_enrollment': 'sum',
        'adult_enrollment': 'sum',
        'total_enrollment': 'sum',
        'date': ['min', 'max', 'nunique']
    }).reset_index()
    
    alsi_data.columns = [
        'state', 'district', 'child_enroll', 'adult_enroll', 'total_enroll',
        'first_date', 'last_date', 'active_days'
    ]
    
    # Days span for lifecycle calculation
    alsi_data['days_span'] = (alsi_data['last_date'] - alsi_data['first_date']).dt.days + 1
    
    # ALSI Formula
    alsi_data['ALSI'] = (
        (alsi_data['child_enroll'] * alsi_data['days_span']) /
        (alsi_data['adult_enroll'] + 1)
    )
    
    # Normalize ALSI
    if len(alsi_data) > 0 and alsi_data['ALSI'].max() > alsi_data['ALSI'].min():
        alsi_data['ALSI_normalized'] = (
            (alsi_data['ALSI'] - alsi_data['ALSI'].min()) /
            (alsi_data['ALSI'].max() - alsi_data['ALSI'].min() + EPSILON) * 100
        )
    else:
        alsi_data['ALSI_normalized'] = 50
    
    # Forecast MBU load for 2030
    alsi_data['MBU_2030_forecast'] = alsi_data['child_enroll']
    
    # Classify ALSI
    alsi_data['ALSI_category'] = pd.cut(
        alsi_data['ALSI_normalized'],
        bins=[0, 25, 50, 75, 100],
        labels=['Low Risk', 'Moderate', 'High Risk', 'Critical'],
        include_lowest=True
    )
    
    return alsi_data


# =============================================================================
# ANOMALY DETECTION
# =============================================================================

def detect_anomalies(df_unified: pd.DataFrame) -> pd.DataFrame:
    """
    Detect anomalies using Isolation Forest and statistical methods.
    Identifies potential ghost centers and fraud patterns.
    """
    df = df_unified.copy()
    
    # Security-based anomaly flags
    security_ratio = df.get('security_ratio', pd.Series([1] * len(df)))
    
    def classify_security(ratio):
        if ratio >= 0.8: return '🟢 Healthy'
        elif ratio >= 0.5: return '🟡 Warning'
        elif ratio >= 0.3: return '🔴 Critical'
        else: return '⚫ Emergency'
    
    df['security_status'] = security_ratio.apply(classify_security)
    
    # Migration anomaly flags
    migration_intensity = df.get('migration_intensity', pd.Series([0] * len(df)))
    
    def classify_migration(intensity):
        if intensity <= 2: return '🟢 Normal'
        elif intensity <= 5: return '🟡 Elevated'
        elif intensity <= 10: return '🔴 High'
        else: return '⚫ Extreme'
    
    df['migration_status'] = migration_intensity.apply(classify_migration)
    
    # Dual Threat Matrix
    df['threat_quadrant'] = 'Normal'
    df.loc[(security_ratio < 0.5) & (migration_intensity <= 5), 'threat_quadrant'] = 'Security Issue'
    df.loc[(security_ratio >= 0.5) & (migration_intensity > 5), 'threat_quadrant'] = 'Migration Issue'
    df.loc[(security_ratio < 0.5) & (migration_intensity > 5), 'threat_quadrant'] = '🚨 DUAL THREAT'
    
    # Aadhaar Integrity Score (AIS)
    def security_penalty(ratio):
        if ratio < 0.1: return 50
        elif ratio < 0.3: return 40
        elif ratio < 0.5: return 30
        elif ratio < 0.8: return 15
        else: return 0
    
    def migration_penalty(intensity):
        if intensity > 10: return 30
        elif intensity > 5: return 20
        elif intensity > 2: return 10
        else: return 0
    
    df['security_penalty'] = security_ratio.apply(security_penalty)
    df['migration_penalty'] = migration_intensity.apply(migration_penalty)
    df['AIS_score'] = (100 - df['security_penalty'] - df['migration_penalty']).clip(0, 100)
    
    def classify_ais(score):
        if score >= 80: return '🟢 Excellent'
        elif score >= 60: return '🟡 Good'
        elif score >= 40: return '🟠 Moderate'
        elif score >= 20: return '🔴 Poor'
        else: return '⚫ Critical'
    
    df['AIS_category'] = df['AIS_score'].apply(classify_ais)
    
    return df


# =============================================================================
# GHOST CENTER DETECTION
# =============================================================================

def detect_ghost_centers(df_bio: pd.DataFrame) -> pd.DataFrame:
    """
    Use ML to detect potentially fraudulent/inactive centers.
    """
    if df_bio is None or len(df_bio) == 0:
        return pd.DataFrame()
    
    # Aggregate at pincode level
    pincode_stats = df_bio.groupby(['state', 'district', 'pincode']).agg({
        'total_bio': ['sum', 'mean', 'std', 'count']
    }).reset_index()
    pincode_stats.columns = ['state', 'district', 'pincode', 'total_bio', 'mean_bio', 'std_bio', 'active_days']
    
    # Fill NaN std with 0
    pincode_stats['std_bio'] = pincode_stats['std_bio'].fillna(0)
    
    # Calculate ghost score components
    national_avg = pincode_stats['total_bio'].mean()
    
    # Activity gap (low activity = suspicious)
    pincode_stats['activity_gap'] = np.clip(
        (national_avg - pincode_stats['total_bio']) / (national_avg + EPSILON), 0, 1
    )
    
    # Variance anomaly
    mean_bio_safe = pincode_stats['mean_bio'].replace(0, EPSILON)
    pincode_stats['variance_anomaly'] = np.clip(
        pincode_stats['std_bio'] / mean_bio_safe / 3, 0, 1
    )
    
    # Isolation Forest
    features = pincode_stats[['total_bio', 'mean_bio', 'active_days']].fillna(0)
    if len(features) > 10:
        iso_forest = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
        pincode_stats['isolation_score'] = (-iso_forest.fit_predict(features) + 1) / 2
    else:
        pincode_stats['isolation_score'] = 0
    
    # Combined ghost score
    pincode_stats['ghost_score'] = (
        0.4 * pincode_stats['activity_gap'] +
        0.3 * pincode_stats['variance_anomaly'] +
        0.3 * pincode_stats['isolation_score']
    )
    
    # Categorize
    pincode_stats['ghost_status'] = pd.cut(
        pincode_stats['ghost_score'],
        bins=[0, 0.4, 0.7, 1.0],
        labels=['Legitimate', 'Suspicious', 'Potential Ghost'],
        include_lowest=True
    )
    
    return pincode_stats


# =============================================================================
# RISK-CAPACITY CLUSTERING
# =============================================================================

def cluster_risk_capacity(df_unified: pd.DataFrame) -> pd.DataFrame:
    """
    Cluster districts into Risk vs Capacity profiles using K-Means.
    Creates 4 governance quadrants:
    - High Risk - Low Capacity (Crisis Zone)
    - High Risk - High Capacity (Overloaded)
    - Low Risk - Low Capacity (Underserved)
    - Low Risk - High Capacity (Champions)
    """
    df = df_unified.copy()
    
    # Risk Score
    df['risk_score'] = df.get('AESI', pd.Series([50] * len(df))).fillna(50)
    
    # Capacity Score
    total_activity = df.get('total_activity', pd.Series([0] * len(df)))
    max_activity = total_activity.max() if total_activity.max() > 0 else 1
    
    enroll_pincodes = df.get('enroll_pincodes', pd.Series([1] * len(df))).replace(0, 1)
    max_pincodes = enroll_pincodes.max() if enroll_pincodes.max() > 0 else 1
    
    df['capacity_score'] = (
        total_activity / max_activity * 50 +
        enroll_pincodes / max_pincodes * 50
    )
    
    # K-Means clustering
    cluster_features = df[['risk_score', 'capacity_score']].fillna(0)
    scaler = StandardScaler()
    cluster_scaled = scaler.fit_transform(cluster_features)
    
    if len(df) > 4:
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(cluster_scaled)
    else:
        df['cluster'] = 0
    
    # Determine quadrant names based on median thresholds
    risk_median = df['risk_score'].median()
    capacity_median = df['capacity_score'].median()
    
    def assign_quadrant(row):
        high_risk = row['risk_score'] > risk_median
        high_capacity = row['capacity_score'] > capacity_median
        
        if high_risk and not high_capacity:
            return "🔴 HIGH RISK - LOW CAPACITY"
        elif high_risk and high_capacity:
            return "🟠 HIGH RISK - HIGH CAPACITY"
        elif not high_risk and not high_capacity:
            return "🟡 LOW RISK - LOW CAPACITY"
        else:
            return "🟢 LOW RISK - HIGH VOLUME"
    
    df['risk_capacity_profile'] = df.apply(assign_quadrant, axis=1)
    
    return df


# =============================================================================
# MAIN RISK ENGINE CLASS
# =============================================================================

class RiskEngine:
    """Main risk calculation engine for UIDAI dashboard."""
    
    def __init__(self):
        self.df_unified = pd.DataFrame()
        self.df_alsi = pd.DataFrame()
        self.df_ghost_centers = pd.DataFrame()
    
    def calculate_all_indices(
        self,
        df_unified: pd.DataFrame,
        df_enroll: pd.DataFrame = None,
        df_bio: pd.DataFrame = None
    ) -> Dict[str, pd.DataFrame]:
        """Calculate all risk indices."""
        
        # Calculate main indices
        self.df_unified = df_unified.copy()
        self.df_unified = calculate_epi(self.df_unified)
        self.df_unified = calculate_duli(self.df_unified)
        self.df_unified = calculate_brs(self.df_unified)
        self.df_unified = calculate_cari(self.df_unified)
        self.df_unified = calculate_aesi(self.df_unified)
        self.df_unified = detect_anomalies(self.df_unified)
        self.df_unified = cluster_risk_capacity(self.df_unified)
        
        # Calculate ALSI if enrollment data available
        if df_enroll is not None and len(df_enroll) > 0:
            self.df_alsi = calculate_alsi(df_enroll)
        
        # Detect ghost centers if biometric data available
        if df_bio is not None and len(df_bio) > 0:
            self.df_ghost_centers = detect_ghost_centers(df_bio)
        
        return {
            'unified': self.df_unified,
            'alsi': self.df_alsi,
            'ghost_centers': self.df_ghost_centers
        }
    
    def get_high_risk_districts(self, threshold: float = 75) -> pd.DataFrame:
        """Get districts with CARI above threshold."""
        if len(self.df_unified) == 0:
            return pd.DataFrame()
        return self.df_unified[self.df_unified['CARI'] >= threshold]
    
    def get_critical_biometric_districts(self) -> pd.DataFrame:
        """Get districts with Critical BRS."""
        if len(self.df_unified) == 0:
            return pd.DataFrame()
        return self.df_unified[self.df_unified['BRS_category'] == 'Critical']
    
    def get_dual_threat_districts(self) -> pd.DataFrame:
        """Get districts flagged as dual threat."""
        if len(self.df_unified) == 0:
            return pd.DataFrame()
        return self.df_unified[self.df_unified['threat_quadrant'] == '🚨 DUAL THREAT']
    
    def get_state_summary(self) -> pd.DataFrame:
        """Get state-level risk summary."""
        if len(self.df_unified) == 0:
            return pd.DataFrame()
        
        return self.df_unified.groupby('state').agg({
            'CARI': 'mean',
            'EPI_normalized': 'mean',
            'DULI_normalized': 'mean',
            'BRS_normalized': 'mean',
            'AESI': 'mean',
            'total_enrollment': 'sum',
            'total_demo': 'sum',
            'total_bio': 'sum',
            'district': 'count'
        }).reset_index().rename(columns={'district': 'district_count'})
    
    def get_regional_summary(self) -> pd.DataFrame:
        """Get regional risk summary."""
        if len(self.df_unified) == 0:
            return pd.DataFrame()
        
        return self.df_unified.groupby('region').agg({
            'CARI': 'mean',
            'EPI_normalized': 'mean',
            'DULI_normalized': 'mean',
            'BRS_normalized': 'mean',
            'AESI': 'mean',
            'total_enrollment': 'sum',
            'total_activity': 'sum',
            'district': 'count',
            'state': 'nunique'
        }).reset_index().rename(columns={
            'district': 'district_count',
            'state': 'state_count'
        })
    
    def generate_policy_insights(self) -> List[Dict]:
        """Generate policy intelligence insights."""
        insights = []
        
        if len(self.df_unified) == 0:
            return insights
        
        # High risk districts insight
        high_risk = self.get_high_risk_districts(75)
        if len(high_risk) > 0:
            insights.append({
                'type': 'warning',
                'title': 'Critical Risk Districts Identified',
                'message': f"{len(high_risk)} districts have CARI > 75, indicating combined stress across enrollment, demographics, and biometrics.",
                'recommendation': "Prioritize infrastructure upgrades and fraud audits in these districts."
            })
        
        # Biometric compliance insight
        low_security = self.df_unified[self.df_unified.get('security_ratio', pd.Series([1]*len(self.df_unified))) < 0.5]
        if len(low_security) > 0:
            insights.append({
                'type': 'alert',
                'title': 'Low Biometric Compliance Detected',
                'message': f"{len(low_security)} districts have security ratio < 0.5, indicating potential operators bypassing biometric verification.",
                'recommendation': "Deploy audit teams and mandate biometric verification compliance training."
            })
        
        # Dual threat insight
        dual_threat = self.get_dual_threat_districts()
        if len(dual_threat) > 0:
            insights.append({
                'type': 'critical',
                'title': 'Dual Threat Districts',
                'message': f"{len(dual_threat)} districts show both security issues AND migration anomalies.",
                'recommendation': "Immediate field investigation and suspension of suspected operators."
            })
        
        # MBU forecast insight
        if len(self.df_alsi) > 0:
            high_alsi = self.df_alsi[self.df_alsi['ALSI_normalized'] > 75]
            if len(high_alsi) > 0:
                total_mbu_forecast = high_alsi['MBU_2030_forecast'].sum()
                insights.append({
                    'type': 'forecast',
                    'title': 'Future MBU Crisis Warning',
                    'message': f"{len(high_alsi)} districts at risk of MBU backlog by 2030. Projected {total_mbu_forecast:,.0f} children requiring updates.",
                    'recommendation': "Begin capacity building in these districts immediately."
                })
        
        return insights
