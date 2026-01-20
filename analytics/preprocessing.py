"""
================================================================================
UIDAI GOVERNANCE INTELLIGENCE DASHBOARD - DATA PREPROCESSING MODULE
================================================================================
Handles data loading, transformation, and index computation for:
- AESI (Aadhaar Enrollment Stress Index)
- BUSI (Biometric Update Stress Index)
- ALSI (Aadhaar Lifecycle Stress Index)
- DULI (Demographic Update Load Index)
- Ghost Score (Anomaly Detection)
- Leaderboard Score (Performance Ranking)

SHARED MODULE: Used by both notebook and dashboard for analytical consistency.
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st
from typing import Dict, Tuple, Optional


# =============================================================================
# STATE MAPPING (Cities incorrectly in state column → Canonical State)
# =============================================================================

STATE_MAPPING = {
    # Cities/Districts incorrectly in state column
    'Balanagar': 'Telangana', 'BALANAGAR': 'Telangana', 'balanagar': 'Telangana',
    'Darbhanga': 'Bihar', 'DARBHANGA': 'Bihar', 'darbhanga': 'Bihar',
    'Jaipur': 'Rajasthan', 'JAIPUR': 'Rajasthan', 'jaipur': 'Rajasthan',
    'Madanapalle': 'Andhra Pradesh', 'MADANAPALLE': 'Andhra Pradesh',
    'Nagpur': 'Maharashtra', 'NAGPUR': 'Maharashtra', 'nagpur': 'Maharashtra',
    'Puttenahalli': 'Karnataka', 'PUTTENAHALLI': 'Karnataka',
    'Raja Annamalai Puram': 'Tamil Nadu', 'RAJA ANNAMALAI PURAM': 'Tamil Nadu',
    
    # Odisha variations
    'Orissa': 'Odisha', 'ODISHA': 'Odisha', 'odisha': 'Odisha',
    
    # West Bengal variations
    'WEST BENGAL': 'West Bengal', 'WESTBENGAL': 'West Bengal', 'West  Bengal': 'West Bengal',
    'West Bangal': 'West Bengal', 'West Bengli': 'West Bengal', 'Westbengal': 'West Bengal',
    'West bengal': 'West Bengal', 'west bengal': 'West Bengal', 'west Bengal': 'West Bengal',
    
    # Jammu & Kashmir
    'Jammu & Kashmir': 'Jammu and Kashmir', 'Jammu And Kashmir': 'Jammu and Kashmir',
    'J&K': 'Jammu and Kashmir', 'JAMMU AND KASHMIR': 'Jammu and Kashmir',
    
    # Andaman variations
    'Andaman & Nicobar Islands': 'Andaman and Nicobar Islands',
    'Andaman and Nicobar': 'Andaman and Nicobar Islands',
    'A&N Islands': 'Andaman and Nicobar Islands',
    
    # Chhattisgarh
    'Chhatisgarh': 'Chhattisgarh', 'CHHATTISGARH': 'Chhattisgarh',
    
    # Puducherry
    'Pondicherry': 'Puducherry', 'PUDUCHERRY': 'Puducherry',
    
    # Tamil Nadu
    'Tamilnadu': 'Tamil Nadu', 'TAMIL NADU': 'Tamil Nadu',
    
    # Uttarakhand
    'Uttaranchal': 'Uttarakhand', 'UTTARAKHAND': 'Uttarakhand',
    
    # Dadra and Nagar Haveli
    'Dadra & Nagar Haveli': 'Dadra and Nagar Haveli and Daman and Diu',
    'Dadra and Nagar Haveli': 'Dadra and Nagar Haveli and Daman and Diu',
    'Daman & Diu': 'Dadra and Nagar Haveli and Daman and Diu',
    'Daman and Diu': 'Dadra and Nagar Haveli and Daman and Diu',
    'The Dadra And Nagar Haveli And Daman And Diu': 'Dadra and Nagar Haveli and Daman and Diu',
    
    # Delhi
    'Delhi': 'NCT of Delhi', 'DELHI': 'NCT of Delhi', 'New Delhi': 'NCT of Delhi',
    
    # Other case variations
    'TELANGANA': 'Telangana', 'MAHARASHTRA': 'Maharashtra', 'KARNATAKA': 'Karnataka',
    'GUJARAT': 'Gujarat', 'RAJASTHAN': 'Rajasthan', 'KERALA': 'Kerala',
    'ANDHRA PRADESH': 'Andhra Pradesh', 'andhra pradesh': 'Andhra Pradesh',
    'UTTAR PRADESH': 'Uttar Pradesh', 'MADHYA PRADESH': 'Madhya Pradesh',
    'BIHAR': 'Bihar', 'JHARKHAND': 'Jharkhand', 'ASSAM': 'Assam',
    'PUNJAB': 'Punjab', 'HARYANA': 'Haryana',
}

# =============================================================================
# DISTRICT NAME MAPPING (Variant → Canonical Name)
# Mirrors UIDAI_HACKATHON_FINAL_SUBMISSION.ipynb for analytical consistency
# =============================================================================

DISTRICT_MAPPING = {
    # ---------------- ANDHRA PRADESH ----------------
    'Ananthapur': 'Anantapur',
    'Ananthapuramu': 'Anantapur',
    'Cuddapah': 'Y.S.R.',
    'Y. S. R': 'Y.S.R.',
    'Visakhapatanam': 'Visakhapatnam',
    'Spsr Nellore': 'Sri Potti Sriramulu Nellore',
    'Nellore': 'Sri Potti Sriramulu Nellore',
    'Mahabub Nagar': 'Mahabubnagar',
    'Mahbubnagar': 'Mahabubnagar',
    'Karim Nagar': 'Karimnagar',

    # ---------------- BIHAR ----------------
    'Aurangabad(BH)': 'Aurangabad',
    'Aurangabad(bh)': 'Aurangabad',
    'Monghyr': 'Munger',
    'Purnea': 'Purnia',
    'Samstipur': 'Samastipur',
    'Sheikpura': 'Sheikhpura',
    'East Champaran': 'Purbi Champaran',
    'Pashchim Champaran': 'West Champaran',
    'Bhabua': 'Kaimur',

    # ---------------- HIMACHAL PRADESH ----------------
    'Lahaul and Spiti': 'Lahul and Spiti',
    'Lahul  Spiti': 'Lahul and Spiti',

    # ---------------- JHARKHAND ----------------
    'Hazaribag': 'Hazaribagh',
    'Purbi Singhbhum': 'East Singhbhum',
    'Pashchimi Singhbhum': 'West Singhbhum',

    # ---------------- KARNATAKA ----------------
    'Belgaum': 'Belagavi',
    'Bellary': 'Ballari',
    'Gulbarga': 'Kalaburagi',
    'Mysore': 'Mysuru',
    'Shimoga': 'Shivamogga',
    'Tumkur': 'Tumakuru',
    'Hasan': 'Hassan',
    'Chamarajanagar': 'Chamrajnagar',
    'Bangalore': 'Bengaluru Urban',
    'Bengaluru': 'Bengaluru Urban',

    # ---------------- MADHYA PRADESH ----------------
    'Hoshangabad': 'Narmadapuram',
    'Narsimhapur': 'Narsinghpur',
    'East Nimar': 'Khandwa',
    'West Nimar': 'Khargone',

    # ---------------- MAHARASHTRA ----------------
    'Ahmadnagar': 'Ahilyanagar',
    'Ahmednagar': 'Ahilyanagar',
    'Ahmed Nagar': 'Ahilyanagar',
    'Aurangabad': 'Chhatrapati Sambhajinagar',
    'Osmanabad': 'Dharashiv',
    'Bid': 'Beed',

    # ---------------- ODISHA ----------------
    'Anugul': 'Angul',
    'Anugal': 'Angul',
    'Nabarangapur': 'Nabarangpur',
    'Sonapur': 'Subarnapur',

    # ---------------- RAJASTHAN ----------------
    'Jalor': 'Jalore',
    'Jhunjhunun': 'Jhunjhunu',
    'Chittaurgarh': 'Chittorgarh',

    # ---------------- TAMIL NADU ----------------
    'Thoothukkudi': 'Thoothukkudi',
    'Tuticorin': 'Thoothukkudi',
    'Thiruvallur': 'Tiruvallur',

    # ---------------- TELANGANA ----------------
    'Jangaon': 'Jangoan',
    'Ranga Reddy': 'Rangareddy',
    'Medchal?malkajgiri': 'Medchal-Malkajgiri',
    'Medchalâmalkajgiri': 'Medchal-Malkajgiri',
    'Medchal−malkajgiri': 'Medchal-Malkajgiri',

    # ---------------- UTTAR PRADESH ----------------
    'Allahabad': 'Prayagraj',
    'Faizabad': 'Ayodhya',
    'Bara Banki': 'Barabanki',
    'Shravasti': 'Shrawasti',

    # ---------------- WEST BENGAL ----------------
    'Burdwan': 'Bardhaman',
    'Hooghiy': 'Hooghly',
    'Hugli': 'Hooghly',
    'Medinipur West': 'Paschim Medinipur',
    'West Midnapore': 'Paschim Medinipur',
    'Puruliya': 'Purulia',
    'South Twenty Four Parganas': 'South 24 Parganas',
    'South 24 Pargana': 'South 24 Parganas',
    'Dinajpur Dakshin': 'Dakshin Dinajpur',
}

# Garbage values to filter out
INVALID_STATE_VALUES = {'100000', '100000 district', '', '?', 'nan', 'none', 'null'}
INVALID_DISTRICT_VALUES = {'100000', '100000 District', '', '?', 'None', 'Na', 'N/A',
                           '5Th Cross', 'Near University Thana', 'Near Meera Hospital',
                           'Idpl Colony', 'Near Dhyana Ashram', 'Near Uday Nagar Nit Garden',
                           'nan', 'none', 'null'}

# Canonical 36 States/UTs
CANONICAL_STATES = {
    'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh',
    'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand', 'Karnataka',
    'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram',
    'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu',
    'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal',
    'Andaman and Nicobar Islands', 'Chandigarh', 'Dadra and Nagar Haveli and Daman and Diu',
    'NCT of Delhi', 'Jammu and Kashmir', 'Ladakh', 'Lakshadweep', 'Puducherry'
}

# =============================================================================
# REGION CLASSIFICATION
# =============================================================================

REGION_MAPPING = {
    'North': ['NCT of Delhi', 'Haryana', 'Himachal Pradesh', 'Jammu and Kashmir', 'Punjab', 
              'Rajasthan', 'Uttarakhand', 'Uttar Pradesh', 'Chandigarh', 'Ladakh'],
    'South': ['Andhra Pradesh', 'Karnataka', 'Kerala', 'Tamil Nadu', 'Telangana', 'Puducherry', 
              'Lakshadweep', 'Andaman and Nicobar Islands'],
    'East': ['Bihar', 'Jharkhand', 'Odisha', 'West Bengal'],
    'West': ['Goa', 'Gujarat', 'Maharashtra', 'Dadra and Nagar Haveli and Daman and Diu'],
    'Central': ['Chhattisgarh', 'Madhya Pradesh'],
    'Northeast': ['Arunachal Pradesh', 'Assam', 'Manipur', 'Meghalaya', 'Mizoram', 
                  'Nagaland', 'Sikkim', 'Tripura']
}


def get_region(state: str) -> str:
    """Classify state into region."""
    for region, states in REGION_MAPPING.items():
        if state in states:
            return region
    return 'Other'


def normalize_state(state_name) -> Optional[str]:
    """Normalize state name to canonical form. Returns None for invalid."""
    if state_name is None or pd.isna(state_name):
        return None
    
    state_str = str(state_name).strip()
    
    # Check if invalid
    if state_str.lower() in INVALID_STATE_VALUES or state_str.isdigit():
        return None
    
    # Check mapping first
    if state_str in STATE_MAPPING:
        return STATE_MAPPING[state_str]
    
    # Try title case
    title_case = state_str.title()
    if title_case in STATE_MAPPING:
        return STATE_MAPPING[title_case]
    
    # Check if already canonical
    if title_case in CANONICAL_STATES:
        return title_case
    
    # Fix common patterns
    cleaned = ' '.join(state_str.split()).title()
    cleaned = cleaned.replace(' And ', ' and ')
    if cleaned in CANONICAL_STATES:
        return cleaned
    
    return None


def normalize_district(district_name) -> Optional[str]:
    """Normalize district name. Returns None for garbage values."""
    if district_name is None or pd.isna(district_name):
        return None
    
    # Convert to string and clean
    district_str = str(district_name).strip()
    
    # Remove special characters
    district_str = district_str.replace('*', '').replace('#', '').replace('@', '').strip()
    
    # Check if invalid/garbage
    if district_str.lower() in {v.lower() for v in INVALID_DISTRICT_VALUES} or district_str.isdigit():
        return None
    
    # Title case for standardization
    district_title = ' '.join(district_str.split()).title()
    
    # Check mapping (case-insensitive lookup)
    if district_title in DISTRICT_MAPPING:
        return DISTRICT_MAPPING[district_title]
    
    # Try original case in mapping
    if district_str in DISTRICT_MAPPING:
        return DISTRICT_MAPPING[district_str]
    
    # Check lowercase key match
    for key, value in DISTRICT_MAPPING.items():
        if key.lower() == district_str.lower():
            return value
    
    # Return cleaned title case if not in mapping
    return district_title


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

# Division safety constant (same as notebook)
EPSILON = 1e-6


def normalize_series(series: pd.Series, lower: float = 0, upper: float = 100) -> pd.Series:
    """Min-max normalize a series to [lower, upper]."""
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series([50] * len(series), index=series.index)
    return lower + (series - min_val) / (max_val - min_val) * (upper - lower)


def load_csv_files(data_dir: Path, pattern: str, use_sampling: bool = True, sample_size: int = 50000) -> pd.DataFrame:
    """Load and concatenate CSV files matching pattern."""
    files = list(data_dir.glob(pattern))
    if not files:
        return pd.DataFrame()
    
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            if use_sampling and len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {f.name}: {e}")
    
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# =============================================================================
# UNIFIED CLEANING FUNCTION (EXACT MATCH WITH NOTEBOOK)
# =============================================================================

def clean_dataset(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Apply standardized cleaning pipeline to any Aadhaar dataset.
    EXACT MATCH with UIDAI_HACKATHON_FINAL_SUBMISSION.ipynb Section 2.
    """
    
    if df is None or len(df) == 0:
        print(f"   ⚠️ {name}: Empty dataset")
        return pd.DataFrame()
    
    df_clean = df.copy()
    initial_count = len(df_clean)
    
    # Standardize column names
    df_clean.columns = df_clean.columns.str.lower().str.strip()
    
    # Step 1: Date conversion
    if 'date' in df_clean.columns:
        df_clean['date'] = pd.to_datetime(df_clean['date'], format='mixed', errors='coerce')
    elif 'registrationdate' in df_clean.columns:
        df_clean['date'] = pd.to_datetime(df_clean['registrationdate'], format='mixed', errors='coerce')
    
    invalid_dates = df_clean['date'].isna().sum() if 'date' in df_clean.columns else 0
    df_clean = df_clean.dropna(subset=['date']) if 'date' in df_clean.columns else df_clean
    
    # Get raw state/district columns
    for col in ['state', 'district', 'state_name', 'district_name']:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip()
    
    if 'state_name' in df_clean.columns and 'state' not in df_clean.columns:
        df_clean['state'] = df_clean['state_name']
    if 'district_name' in df_clean.columns and 'district' not in df_clean.columns:
        df_clean['district'] = df_clean['district_name']
    
    # Step 2: State normalization (ensures 36 canonical states)
    if 'state' in df_clean.columns:
        df_clean['state'] = df_clean['state'].apply(normalize_state)
        invalid_states = df_clean['state'].isna().sum()
        df_clean = df_clean.dropna(subset=['state'])
    else:
        invalid_states = 0
    
    # Step 3: District normalization (canonical naming)
    if 'district' in df_clean.columns:
        df_clean['district'] = df_clean['district'].apply(normalize_district)
        invalid_districts = df_clean['district'].isna().sum()
        df_clean = df_clean.dropna(subset=['district'])
    else:
        invalid_districts = 0
    
    # Step 4: Remove duplicates
    before_dedup = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    duplicates_removed = before_dedup - len(df_clean)
    
    # Step 5: Fill numeric NaNs with 0
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(0)
    
    # Step 6: Add temporal & regional features
    if 'state' in df_clean.columns:
        df_clean['region'] = df_clean['state'].apply(get_region)
    
    if 'date' in df_clean.columns:
        df_clean['year'] = df_clean['date'].dt.year
        df_clean['month'] = df_clean['date'].dt.month
        df_clean['month_name'] = df_clean['date'].dt.month_name()
        df_clean['day_of_week'] = df_clean['date'].dt.dayofweek
        df_clean['day_name'] = df_clean['date'].dt.day_name()
        df_clean['is_weekend'] = df_clean['day_of_week'].isin([5, 6]).astype(int)
        df_clean['quarter'] = df_clean['date'].dt.quarter
        df_clean['week_of_year'] = df_clean['date'].dt.isocalendar().week
    
    final_count = len(df_clean)
    
    print(f"""
   📊 {name} Cleaning Report:
   ──────────────────────────────────────────
   - Initial records:      {initial_count:>12,}
   - Invalid dates:        {invalid_dates:>12,}
   - Invalid states:       {invalid_states:>12,}
   - Invalid districts:    {invalid_districts:>12,}
   - Duplicates removed:   {duplicates_removed:>12,}
   - Final clean records:  {final_count:>12,}
   - Data retention rate:  {final_count/initial_count*100:>11.1f}%
""")
    return df_clean


@st.cache_resource(ttl=3600)  # SHARED across all users for memory efficiency
def load_enrolment_data(data_dir: str, use_sampling: bool = True) -> pd.DataFrame:
    """
    Load and clean enrollment data from CSV files.
    EXACT MATCH with UIDAI_HACKATHON_FINAL_SUBMISSION.ipynb preprocessing.
    """
    base = Path(data_dir) / "api_data_aadhar_enrolment"
    df = load_csv_files(base, "*.csv", use_sampling)
    
    if len(df) == 0:
        return df
    
    # Apply unified cleaning (matches notebook Section 2)
    df = clean_dataset(df, 'ENROLLMENT')
    
    if len(df) == 0:
        return df
    
    # Create enrollment totals (EXACT MATCH with notebook Section 2.4)
    enrollment_cols = ['age_0_5', 'age_5_17', 'age_18_greater']
    for col in enrollment_cols:
        if col not in df.columns:
            df[col] = 0
    
    # Total enrollment metrics
    df['total_enrollment'] = df['age_0_5'] + df['age_5_17'] + df['age_18_greater']
    df['child_enrollment'] = df['age_0_5'] + df['age_5_17']
    df['adult_enrollment'] = df['age_18_greater']
    
    # Child-to-adult ratio (indicator of demographic skew)
    df['child_adult_ratio'] = df['child_enrollment'] / (df['adult_enrollment'] + EPSILON)
    
    return df


@st.cache_resource(ttl=3600)  # SHARED across all users for memory efficiency
def load_biometric_data(data_dir: str, use_sampling: bool = True) -> pd.DataFrame:
    """
    Load and clean biometric data from CSV files.
    EXACT MATCH with UIDAI_HACKATHON_FINAL_SUBMISSION.ipynb preprocessing.
    """
    base = Path(data_dir) / "api_data_aadhar_biometric" / "api_data_aadhar_biometric"
    if not base.exists():
        base = Path(data_dir) / "api_data_aadhar_biometric"
    
    df = load_csv_files(base, "*.csv", use_sampling)
    
    if len(df) == 0:
        return df
    
    # Apply unified cleaning (matches notebook Section 2)
    df = clean_dataset(df, 'BIOMETRIC')
    
    if len(df) == 0:
        return df
    
    # Rename columns to match notebook convention
    col_renames = {'bio_age_5_17': 'bio_child', 'bio_age_17_': 'bio_adult'}
    for old, new in col_renames.items():
        if old in df.columns:
            df = df.rename(columns={old: new})
    
    if 'bio_child' not in df.columns:
        df['bio_child'] = 0
    if 'bio_adult' not in df.columns:
        df['bio_adult'] = 0
    
    # Create biometric totals (EXACT MATCH with notebook)
    df['total_biometric'] = df['bio_child'] + df['bio_adult']
    
    # MBU Compliance Index: Child biometric updates relative to adult
    df['mbu_compliance_index'] = df['bio_child'] / (df['bio_adult'] + EPSILON)
    
    return df


@st.cache_resource(ttl=3600)  # SHARED across all users for memory efficiency
def load_demographic_data(data_dir: str, use_sampling: bool = True) -> pd.DataFrame:
    """
    Load and clean demographic update data from CSV files.
    EXACT MATCH with UIDAI_HACKATHON_FINAL_SUBMISSION.ipynb preprocessing.
    """
    base = Path(data_dir) / "api_data_aadhar_demographic"
    df = load_csv_files(base, "*.csv", use_sampling)
    
    if len(df) == 0:
        return df
    
    # Apply unified cleaning (matches notebook Section 2)
    df = clean_dataset(df, 'DEMOGRAPHIC')
    
    if len(df) == 0:
        return df
    
    # Handle duplicate columns
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    
    # Create demo_child column (EXACT MATCH with notebook)
    if 'demo_age_5_17' in df.columns:
        df['demo_child'] = df['demo_age_5_17'].values
    else:
        df['demo_child'] = 0
    
    # Create demo_adult column
    if 'demo_age_17_' in df.columns:
        df['demo_adult'] = df['demo_age_17_'].values
    else:
        df['demo_adult'] = 0
    
    # Create total_demo (EXACT MATCH with notebook)
    df['total_demo'] = df['demo_child'] + df['demo_adult']
    
    return df


@st.cache_resource(ttl=3600)  # SHARED across all users for memory efficiency
def build_risk_dataframe(_df_enrol, _df_bio, _df_demo):
    """Build unified risk dataframe with all computed indices."""
    
    # Aggregate by state-district
    group_cols = ['state', 'district']
    
    # Enrollment aggregation
    enrol_agg = pd.DataFrame()
    if len(_df_enrol) > 0 and all(c in _df_enrol.columns for c in group_cols):
        enrol_agg = _df_enrol.groupby(group_cols).agg({
            'total_enrollment': 'sum'
        }).reset_index()
        
        # Add center count if available
        if 'enrolment_agency' in _df_enrol.columns or 'agency' in _df_enrol.columns:
            agency_col = 'enrolment_agency' if 'enrolment_agency' in _df_enrol.columns else 'agency'
            center_counts = _df_enrol.groupby(group_cols)[agency_col].nunique().reset_index()
            center_counts.columns = ['state', 'district', 'center_count']
            enrol_agg = enrol_agg.merge(center_counts, on=group_cols, how='left')
        else:
            enrol_agg['center_count'] = 1
    
    # Biometric aggregation
    bio_agg = pd.DataFrame()
    if len(_df_bio) > 0 and all(c in _df_bio.columns for c in group_cols):
        bio_agg = _df_bio.groupby(group_cols).agg({
            'total_biometric': 'sum'
        }).reset_index()
    
    # Demographic aggregation
    demo_agg = pd.DataFrame()
    if len(_df_demo) > 0 and all(c in _df_demo.columns for c in group_cols):
        demo_agg = _df_demo.groupby(group_cols).agg({
            'total_demo': 'sum'
        }).reset_index()
    
    # Merge all
    if len(enrol_agg) == 0:
        return pd.DataFrame()
    
    risk_df = enrol_agg.copy()
    if len(bio_agg) > 0:
        risk_df = risk_df.merge(bio_agg, on=group_cols, how='outer')
    if len(demo_agg) > 0:
        risk_df = risk_df.merge(demo_agg, on=group_cols, how='outer')
    
    # Fill missing values
    risk_df = risk_df.fillna(0)
    
    # Compute total activity
    risk_df['total_activity'] = (
        risk_df.get('total_enrollment', 0) + 
        risk_df.get('total_biometric', 0) + 
        risk_df.get('total_demo', 0)
    )
    
    # Ensure center_count > 0
    if 'center_count' not in risk_df.columns:
        risk_df['center_count'] = 1
    risk_df['center_count'] = risk_df['center_count'].replace(0, 1)
    
    # Compute AESI (Aadhaar Enrollment Stress Index)
    risk_df['enrollment_per_center'] = risk_df['total_enrollment'] / risk_df['center_count']
    risk_df['EPI'] = normalize_series(risk_df['enrollment_per_center'])
    risk_df['CARI'] = normalize_series(risk_df['center_count'].apply(lambda x: 100 - x))
    risk_df['AESI'] = 0.6 * risk_df['EPI'] + 0.4 * risk_df['CARI']
    
    # Compute BUSI (Biometric Update Stress Index)
    if 'total_biometric' in risk_df.columns:
        risk_df['biometric_per_center'] = risk_df['total_biometric'] / risk_df['center_count']
        risk_df['BUSI'] = normalize_series(risk_df['biometric_per_center'])
    else:
        risk_df['BUSI'] = 0
    
    # Compute DULI (Demographic Update Load Index)
    if 'total_demo' in risk_df.columns:
        risk_df['demo_per_center'] = risk_df['total_demo'] / risk_df['center_count']
        risk_df['DULI'] = normalize_series(risk_df['demo_per_center'])
        risk_df['DULI_normalized'] = risk_df['DULI']
    else:
        risk_df['DULI'] = 0
        risk_df['DULI_normalized'] = 0
    
    # Compute ALSI (Aadhaar Lifecycle Stress Index)
    risk_df['ALSI'] = 0.5 * risk_df['AESI'] + 0.3 * risk_df['BUSI'] + 0.2 * risk_df['DULI']
    
    # Ghost Score (simple version - high activity + low centers)
    risk_df['Ghost_Score'] = normalize_series(risk_df['total_activity'] / risk_df['center_count'])
    
    # Risk categories
    def categorize_risk(score):
        if score < 25: return 'Low'
        elif score < 50: return 'Moderate'
        elif score < 75: return 'High'
        else: return 'Critical'
    
    risk_df['risk_category'] = risk_df['AESI'].apply(categorize_risk)
    
    # Leaderboard Score (inverse of stress - lower is better)
    risk_df['Leaderboard_Score'] = 100 - (0.4 * risk_df['AESI'] + 0.3 * risk_df['BUSI'] + 0.3 * risk_df['DULI'])
    
    # Risk and Capacity scores for quadrant analysis
    risk_df['Risk_Score'] = risk_df['AESI']
    risk_df['Capacity_Score'] = normalize_series(risk_df['center_count'])
    
    return risk_df


def get_state_mapping() -> Dict[str, str]:
    """Return state name to code mapping for India choropleth."""
    return {
        'Andhra Pradesh': 'AP', 'Arunachal Pradesh': 'AR', 'Assam': 'AS',
        'Bihar': 'BR', 'Chhattisgarh': 'CT', 'Goa': 'GA',
        'Gujarat': 'GJ', 'Haryana': 'HR', 'Himachal Pradesh': 'HP',
        'Jharkhand': 'JH', 'Karnataka': 'KA', 'Kerala': 'KL',
        'Madhya Pradesh': 'MP', 'Maharashtra': 'MH', 'Manipur': 'MN',
        'Meghalaya': 'ML', 'Mizoram': 'MZ', 'Nagaland': 'NL',
        'Odisha': 'OR', 'Punjab': 'PB', 'Rajasthan': 'RJ',
        'Sikkim': 'SK', 'Tamil Nadu': 'TN', 'Telangana': 'TG',
        'Tripura': 'TR', 'Uttar Pradesh': 'UP', 'Uttarakhand': 'UT',
        'West Bengal': 'WB', 'Delhi': 'DL', 'Jammu And Kashmir': 'JK',
        'Ladakh': 'LA', 'Puducherry': 'PY', 'Chandigarh': 'CH',
        'Andaman And Nicobar Islands': 'AN', 'Dadra And Nagar Haveli And Daman And Diu': 'DN',
        'Lakshadweep': 'LD'
    }
