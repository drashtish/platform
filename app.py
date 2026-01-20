"""
================================================================================
UIDAI GOVERNANCE INTELLIGENCE DASHBOARD
================================================================================
Production-grade decision-support platform for UIDAI operations.
Mirrors UIDAI_HACKATHON_FINAL_SUBMISSION.ipynb research notebook.

THREE-LAYER DATA ARCHITECTURE:
    Raw CSV → pipelines/run_pipeline.py → Certified Parquet → Dashboard
    
This dashboard NEVER reads raw CSVs directly. All data must be pre-certified
by the pipeline to ensure analytical integrity and consistency with notebooks.

Key Metrics Implemented:
- AESI (Aadhaar Ecosystem Stress Index)
- ALSI (Aadhaar Lifecycle Stress Index) 
- BUSI (Biometric Update Stress Index)
- AIS (Aadhaar Integrity Score)
- Ghost Center Detection Score
- Risk-Capacity Quadrant Analysis
- Cost-Benefit Analysis
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime
from pathlib import Path

# =============================================================================
# CRITICAL: SELF-INITIALIZING DATA LAYER
# =============================================================================
# This MUST run before any dashboard code to ensure data exists
# Streamlit Cloud containers start fresh - data must be verified/rebuilt
# =============================================================================

from pipelines.bootstrap import ensure_certified_data, check_certified_data_exists, get_data_status

# Early data check - fail gracefully if data unavailable
_data_status = check_certified_data_exists()
if not _data_status["exists"]:
    st.set_page_config(page_title="UIDAI - Initializing", page_icon="⏳", layout="wide")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0c1929 0%, #1a2942 100%); 
                padding: 60px; border-radius: 16px; text-align: center; 
                border: 2px solid #d4af37; margin-top: 50px;">
        <h1 style="color: #d4af37; margin-bottom: 20px;">🏛️ UIDAI Governance Platform</h1>
        <h2 style="color: #ffffff; margin-bottom: 30px;">System Initializing...</h2>
        <p style="color: #a0aec0; font-size: 18px;">
            First-time setup in progress. Building certified datasets.
        </p>
        <p style="color: #f59e0b; font-size: 16px; margin-top: 20px;">
            ⏱️ Please wait 10-15 seconds and refresh the page.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show progress
    with st.spinner("Building certified data pipeline..."):
        try:
            success = ensure_certified_data(verbose=True)
            if success:
                st.success("✅ Data initialized! Refreshing...")
                time.sleep(2)
                st.rerun()
            else:
                st.error("❌ Data initialization failed. Please contact support.")
                status = get_data_status()
                with st.expander("Technical Details"):
                    st.json(status)
        except Exception as e:
            st.error(f"❌ Initialization error: {str(e)}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())
    
    st.stop()

# =============================================================================
# DATA VERIFIED - PROCEED WITH NORMAL IMPORTS
# =============================================================================

# Import analytics functions
from analytics import (
    load_enrolment_data,
    load_biometric_data,
    load_demographic_data,
    build_risk_dataframe
)

# Import dashboards
from dashboards import (
    render_national_dashboard,
    render_enrolment_dashboard,
    render_biometric_dashboard,
    render_demographic_dashboard
)

# =============================================================================
# PAGE CONFIGURATION - GOVERNMENT GRADE
# =============================================================================

st.set_page_config(
    page_title="UIDAI Governance Intelligence | भारत सरकार",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': """
        **UIDAI Governance Intelligence Dashboard**
        
        Decision Support Platform for Aadhaar Operations
        
        Developed for: Unique Identification Authority of India
        Classification: Internal Use Only
        """
    }
)

# =============================================================================
# GOVERNMENT-GRADE DARK THEME STYLING
# =============================================================================

st.markdown("""
<style>
    /* ========== GOVERNMENT COLOR PALETTE ========== */
    :root {
        --uidai-navy: #0c1929;
        --uidai-navy-light: #1a2942;
        --uidai-navy-hover: #243656;
        --uidai-gold: #d4af37;
        --uidai-white: #ffffff;
        --uidai-gray: #a0aec0;
        --uidai-success: #10b981;
        --uidai-warning: #f59e0b;
        --uidai-danger: #ef4444;
        --uidai-info: #3b82f6;
    }
    
    /* ========== MAIN BODY - DARK NAVY ========== */
    .stApp {
        background: linear-gradient(135deg, #0c1929 0%, #1a2942 50%, #0c1929 100%);
    }
    
    .main > div {
        padding-top: 0.5rem;
    }
    
    /* ========== HIDE DEFAULT ELEMENTS ========== */
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    
    /* Hide header but keep sidebar collapse button visible */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    
    /* Show sidebar collapse/expand button */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"] {
        visibility: visible !important;
        display: block !important;
        background-color: rgba(26, 41, 66, 0.9) !important;
        border-radius: 8px !important;
        border: 1px solid rgba(212, 175, 55, 0.4) !important;
    }
    
    [data-testid="stSidebarCollapseButton"] svg,
    [data-testid="collapsedControl"] svg {
        color: #d4af37 !important;
    }
    
    /* ========== SIDEBAR - INSTITUTIONAL STYLE ========== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1628 0%, #152238 50%, #0a1628 100%);
        border-right: 1px solid rgba(212, 175, 55, 0.3);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
        color: #ecf0f1;
    }
    
    [data-testid="stSidebar"] .stRadio label {
        color: #ecf0f1 !important;
    }
    
    [data-testid="stSidebar"] hr {
        border-color: rgba(212, 175, 55, 0.3);
    }
    
    /* ========== METRIC CARDS - GOVERNMENT STYLE ========== */
    [data-testid="stMetricValue"] {
        font-size: 32px;
        font-weight: 800;
        color: #ffffff;
        font-family: 'Segoe UI', 'Arial', sans-serif;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 14px;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 14px;
    }
    
    /* ========== DATA TABLES - PROFESSIONAL ========== */
    [data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid rgba(212, 175, 55, 0.2);
    }
    
    .stDataFrame thead th {
        background-color: #1a2942 !important;
        color: #d4af37 !important;
        font-weight: 700;
    }
    
    /* ========== HEADERS - GOVERNMENT TYPOGRAPHY ========== */
    h1 {
        font-family: 'Segoe UI', 'Arial', sans-serif;
        color: #ffffff !important;
        font-weight: 800;
        letter-spacing: -0.5px;
        border-bottom: 2px solid #d4af37;
        padding-bottom: 10px;
    }
    
    h2 {
        font-family: 'Segoe UI', 'Arial', sans-serif;
        color: #d4af37 !important;
        font-weight: 700;
    }
    
    h3 {
        font-family: 'Segoe UI', 'Arial', sans-serif;
        color: #ffffff !important;
        font-weight: 600;
    }
    
    /* ========== ALERT BOXES ========== */
    .stAlert {
        border-radius: 8px;
        border-left: 4px solid #d4af37;
    }
    
    /* ========== TABS - GOVERNMENT STYLE (ENHANCED) ========== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(26, 41, 66, 0.9);
        border-radius: 12px;
        padding: 8px;
        border: 1px solid rgba(212, 175, 55, 0.3);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        border-radius: 8px;
        color: #a0aec0;
        font-weight: 600;
        font-size: 14px;
        padding: 0 20px;
        background-color: rgba(12, 25, 41, 0.6);
        border: 1px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(212, 175, 55, 0.2);
        color: #ffffff;
        border: 1px solid rgba(212, 175, 55, 0.4);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #d4af37 0%, #e6c54a 100%) !important;
        color: #0c1929 !important;
    }
    
    /* ========== BUTTONS ========== */
    .stButton button {
        background: linear-gradient(135deg, #d4af37 0%, #b8972d 100%);
        color: #0c1929;
        font-weight: 700;
        border: none;
        border-radius: 6px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #e6c54a 0%, #d4af37 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(212, 175, 55, 0.4);
    }
    
    /* ========== SELECT BOXES ========== */
    .stSelectbox > div > div {
        background-color: rgba(26, 41, 66, 0.8);
        border: 1px solid rgba(212, 175, 55, 0.3);
        color: #ffffff;
    }
    
    /* ========== EXPANDER ========== */
    .streamlit-expanderHeader {
        background-color: rgba(26, 41, 66, 0.8);
        border: 1px solid rgba(212, 175, 55, 0.2);
        border-radius: 8px;
        color: #ffffff;
    }
    
    /* ========== GOVERNMENT HEADER BANNER ========== */
    .gov-banner {
        background: linear-gradient(90deg, #0c1929 0%, #1a2942 50%, #0c1929 100%);
        border-bottom: 3px solid #d4af37;
        padding: 15px 20px;
        margin: -1rem -1rem 1rem -1rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .gov-logo {
        display: flex;
        align-items: center;
        gap: 15px;
    }
    
    .gov-title {
        color: #ffffff;
        font-size: 24px;
        font-weight: 800;
        font-family: 'Segoe UI', sans-serif;
        margin: 0;
    }
    
    .gov-subtitle {
        color: #d4af37;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin: 0;
    }
    
    .gov-timestamp {
        color: #a0aec0;
        font-size: 11px;
        text-align: right;
    }
    
    /* ========== KPI CARD GRID ========== */
    .kpi-card {
        background: linear-gradient(135deg, rgba(26, 41, 66, 0.9) 0%, rgba(12, 25, 41, 0.95) 100%);
        border: 1px solid rgba(212, 175, 55, 0.3);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .kpi-card:hover {
        border-color: #d4af37;
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    }
    
    .kpi-value {
        font-size: 36px;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
    }
    
    .kpi-label {
        font-size: 12px;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 5px;
    }
    
    .kpi-delta {
        font-size: 13px;
        margin-top: 5px;
    }
    
    .kpi-delta.positive { color: #10b981; }
    .kpi-delta.negative { color: #ef4444; }
    .kpi-delta.neutral { color: #f59e0b; }
    
    /* ========== RISK INDICATOR BADGES ========== */
    .risk-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
    }
    
    .risk-critical { background: #ef4444; color: white; }
    .risk-high { background: #f59e0b; color: #0c1929; }
    .risk-moderate { background: #eab308; color: #0c1929; }
    .risk-low { background: #10b981; color: white; }
    
    /* ========== PLOTLY CHART CONTAINER ========== */
    .stPlotlyChart {
        background: rgba(26, 41, 66, 0.5);
        border-radius: 12px;
        padding: 10px;
        border: 1px solid rgba(212, 175, 55, 0.2);
    }
    
    /* ========== SECTION DIVIDERS ========== */
    .section-header {
        background: linear-gradient(90deg, transparent 0%, rgba(212, 175, 55, 0.3) 50%, transparent 100%);
        padding: 10px 20px;
        margin: 20px 0;
        border-radius: 4px;
    }
    
    .section-header h2 {
        margin: 0;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# LIGHT THEME CSS (Applied dynamically based on user selection)
# =============================================================================
def apply_theme():
    """Apply light theme CSS if selected by user."""
    if st.session_state.get('theme') == 'light':
        st.markdown("""
        <style>
            /* ========== LIGHT THEME OVERRIDES ========== */
            .stApp {
                background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 50%, #f5f7fa 100%) !important;
            }
            
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #ffffff 0%, #f0f2f5 50%, #ffffff 100%) !important;
                border-right: 1px solid #d4af37 !important;
            }
            
            [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
            [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
            [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
            [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
            [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h4 {
                color: #1a2942 !important;
            }
            
            h1, h2, h3 { color: #1a2942 !important; }
            p { color: #4a5568 !important; }
            
            [data-testid="stMetricValue"] { color: #1a2942 !important; }
            [data-testid="stMetricLabel"] { color: #4a5568 !important; }
            
            .stTabs [data-baseweb="tab-list"] {
                background-color: rgba(255, 255, 255, 0.9) !important;
                border: 1px solid #d4af37 !important;
            }
            
            .stTabs [data-baseweb="tab"] {
                color: #1a2942 !important;
                background-color: rgba(240, 242, 245, 0.8) !important;
            }
            
            .stDataFrame thead th {
                background-color: #f0f2f5 !important;
                color: #1a2942 !important;
            }
        </style>
        """, unsafe_allow_html=True)





# =============================================================================
# DATA LOADING - CERTIFIED PARQUET ONLY (THREE-LAYER ARCHITECTURE)
# =============================================================================

def get_certified_data_path() -> Path:
    """Return path to certified data directory."""
    # Certified data is in uidai_dashboard/data/certified/
    return Path(__file__).parent / "data" / "certified"


def check_certified_data_exists() -> dict:
    """
    Check if certified parquet files exist and return metadata.
    Returns dict with 'exists' bool and 'metadata' if available.
    """
    certified_path = get_certified_data_path()
    
    required_files = [
        "master_dataset.parquet",
        "enrolment_clean.parquet", 
        "biometric_clean.parquet",
        "demographic_clean.parquet"
    ]
    
    all_exist = all((certified_path / f).exists() for f in required_files)
    
    result = {"exists": all_exist, "path": str(certified_path)}
    
    # Try to load metadata
    metadata_file = certified_path / "pipeline_metadata.json"
    if metadata_file.exists():
        import json
        with open(metadata_file, 'r') as f:
            result["metadata"] = json.load(f)
    
    return result


@st.cache_resource(ttl=3600, show_spinner=False)  # SHARED across all users - critical for multi-user
def load_certified_data():
    """
    Load PRE-CERTIFIED data from parquet files.
    
    THREE-LAYER ARCHITECTURE:
    1. Raw CSV → pipelines/run_pipeline.py → Certified Parquet
    2. Dashboard reads ONLY certified data (this function)
    3. Ensures analytical consistency with notebook analysis
    
    NEVER reads raw CSVs - all data must be pre-certified by pipeline.
    
    NOTE: Using cache_resource to SHARE data across all users.
    This prevents memory exhaustion with multiple concurrent users.
    """
    certified_path = get_certified_data_path()
    
    # Load certified datasets
    df_enrol = pd.read_parquet(certified_path / "enrolment_clean.parquet")
    df_demo = pd.read_parquet(certified_path / "demographic_clean.parquet")
    df_bio = pd.read_parquet(certified_path / "biometric_clean.parquet")
    df_unified = pd.read_parquet(certified_path / "master_dataset.parquet")
    
    return df_enrol, df_demo, df_bio, df_unified


@st.cache_resource(ttl=3600, show_spinner=False)  # SHARED across all users
def calculate_all_risks(_df_enrol, _df_demo, _df_bio, _df_unified):
    """Return the unified risk dataframe (already computed by pipeline)."""
    
    # The unified dataframe already contains all computed indices from pipeline
    return _df_unified


def check_data_freshness(base_path: str) -> tuple:
    """Check modification times of certified data files."""
    
    # Check certified data instead of raw CSVs
    certified_path = get_certified_data_path()
    master_file = certified_path / "master_dataset.parquet"
    
    if master_file.exists():
        return datetime.fromtimestamp(master_file.stat().st_mtime)
    
    return None


# =============================================================================
# SIDEBAR NAVIGATION - GOVERNMENT COMMAND CENTER
# =============================================================================

def render_sidebar():
    """Render the sidebar navigation with government styling."""
    
    with st.sidebar:
        # Government Header
        st.markdown("""
        <div style="text-align: center; padding: 20px 0; border-bottom: 2px solid rgba(212, 175, 55, 0.5); margin-bottom: 20px;">
            <div style="font-size: 40px; margin-bottom: 5px;">🏛️</div>
            <h1 style="color: #ffffff; font-size: 18px; margin: 0; font-weight: 800; letter-spacing: 1px;">
                UIDAI
            </h1>
            <p style="color: #d4af37; font-size: 10px; margin: 5px 0 0 0; text-transform: uppercase; letter-spacing: 2px;">
                Governance Intelligence
            </p>
            <p style="color: #a0aec0; font-size: 9px; margin: 5px 0 0 0;">
                भारत सरकार | Government of India
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        
        # Dashboard selection with icons
        st.markdown("####  COMMAND CENTER")
        
        dashboard = st.radio(
            label="Select Dashboard",
            options=[
                " National Overview",
                " Enrollment Intelligence",
                " Biometric Intelligence",
                " Demographic Intelligence",
                " Policy Intelligence"
            ],
            label_visibility="collapsed"
        )
        
        
        st.markdown("---")
        
        # =================================================================
        # 📊 COMPARE MODE
        # =================================================================
        st.markdown("#### COMPARE MODE")
        compare_mode = st.checkbox("Enable Compare", value=st.session_state.get('compare_mode', False))
        st.session_state['compare_mode'] = compare_mode
        
        if compare_mode:
            st.markdown("<p style='color: #a0aec0; font-size: 11px;'>Select 2 states to compare:</p>", unsafe_allow_html=True)
            
            # Get unique states from session state (will be populated after data loads)
            all_states = st.session_state.get('all_states', ['Loading...'])
            
            compare_state_1 = st.selectbox(
                "State 1",
                options=all_states,
                key="compare_state_1",
                label_visibility="collapsed"
            )
            
            compare_state_2 = st.selectbox(
                "State 2", 
                options=all_states,
                key="compare_state_2",
                index=min(1, len(all_states)-1),
                label_visibility="collapsed"
            )
            
            st.session_state['compare_selections'] = (compare_state_1, compare_state_2)
            
            if compare_state_1 and compare_state_2 and compare_state_1 != compare_state_2:
                st.success(f"Comparing: {compare_state_1} vs {compare_state_2}")
            elif compare_state_1 == compare_state_2:
                st.warning("Select different states")
        
        st.markdown("---")
        
        
        # Control Panel
        st.markdown("#### CONTROLS")
        
        auto_refresh = st.checkbox("Auto-refresh (5 min)", value=True)
        # Note: use_sampling removed - data is pre-certified by pipeline
        use_sampling = True  # Kept for backward compatibility
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Refresh", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        with col2:
            if st.button("Export", use_container_width=True):
                st.session_state['export_mode'] = True
        
        # Add Re-Certify button
        if st.button("Re-Certify Data", use_container_width=True, 
                     help="Run the data certification pipeline"):
            st.session_state['run_pipeline'] = True
            st.rerun()
        
        st.markdown("---")
        
        # Risk Legend with Government styling
        st.markdown("#### RISK SCALE")
        st.markdown("""
        <div style="font-size: 11px; background: rgba(26, 41, 66, 0.8); padding: 12px; border-radius: 8px; border: 1px solid rgba(212, 175, 55, 0.2);">
            <div style="display: flex; align-items: center; margin-bottom: 6px;">
                <span style="width: 12px; height: 12px; background: #10b981; border-radius: 2px; margin-right: 8px;"></span>
                <span style="color: #ffffff;">LOW RISK (0-25)</span>
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 6px;">
                <span style="width: 12px; height: 12px; background: #eab308; border-radius: 2px; margin-right: 8px;"></span>
                <span style="color: #ffffff;">MODERATE (25-50)</span>
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 6px;">
                <span style="width: 12px; height: 12px; background: #f59e0b; border-radius: 2px; margin-right: 8px;"></span>
                <span style="color: #ffffff;">HIGH RISK (50-75)</span>
            </div>
            <div style="display: flex; align-items: center;">
                <span style="width: 12px; height: 12px; background: #ef4444; border-radius: 2px; margin-right: 8px;"></span>
                <span style="color: #ffffff;">CRITICAL (75-100)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Footer
        st.markdown("""
        <div style="text-align: center; padding: 10px 0;">
            <p style="color: #a0aec0; font-size: 9px; margin: 0;">
                UIDAI GOVERNANCE v2.0
            </p>
            <p style="color: #718096; font-size: 8px; margin: 3px 0 0 0;">
                CLASSIFIED • INTERNAL USE ONLY
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    return dashboard, auto_refresh, use_sampling


# =============================================================================
# MAIN APPLICATION - GOVERNMENT COMMAND CENTER
# =============================================================================

def render_header(stats: dict):
    """Render government-style header banner."""
    
    st.markdown(f"""
    <div class="gov-banner">
        <div class="gov-logo">
            <div style="font-size: 36px;">🏛️</div>
            <div>
                <p class="gov-title">UIDAI GOVERNANCE INTELLIGENCE</p>
                <p class="gov-subtitle">Decision Support Platform | भारत सरकार</p>
            </div>
        </div>
        <div class="gov-timestamp">
            <div style="color: #10b981; font-weight: 600;">● LIVE</div>
            <div>{datetime.now().strftime('%d %B %Y | %H:%M:%S IST')}</div>
            <div style="color: #d4af37;">{stats.get('districts_covered', 0)} Districts | {stats.get('states_covered', 0)} States</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main application entry point."""
    
    # Render sidebar and get selections
    selected_dashboard, auto_refresh, use_sampling = render_sidebar()
    
    # Get base path for data
    base_path = str(Path(__file__).parent.parent)
    
    # =================================================================
    # THREE-LAYER ARCHITECTURE CHECK
    # Dashboard ONLY reads from certified parquet files
    # =================================================================
    certified_status = check_certified_data_exists()
    
    # Handle re-certification request from sidebar
    # Pipeline is in uidai_dashboard/pipelines/, not workspace root
    dashboard_path = str(Path(__file__).parent)
    
    if st.session_state.get('run_pipeline', False):
        st.session_state['run_pipeline'] = False
        with st.spinner("Running data certification pipeline..."):
            import subprocess
            import sys
            result = subprocess.run(
                [sys.executable, "pipelines/run_pipeline.py", "--force"],
                cwd=dashboard_path,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                st.success("Data re-certified successfully!")
                st.cache_data.clear()
                time.sleep(1)
                st.rerun()
            else:
                st.error(f"Pipeline failed: {result.stderr}")
    
    if not certified_status["exists"]:
        # Auto-run the certification pipeline when data not found
        st.markdown("""
        <div style="background: rgba(26, 41, 66, 0.9); padding: 40px; border-radius: 12px; text-align: center; border: 1px solid rgba(212, 175, 55, 0.3);">
            <div style="font-size: 48px; margin-bottom: 20px;"></div>
            <h2 style="color: #ffffff; margin-bottom: 10px;">Initializing Data Certification</h2>
            <p style="color: #d4af37; margin-bottom: 20px;">Running three-layer data architecture pipeline...</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("Running data certification pipeline (Raw CSV → Certified Parquet)..."):
            import subprocess
            import sys
            dashboard_path = str(Path(__file__).parent)
            result = subprocess.run(
                [sys.executable, "pipelines/run_pipeline.py", "--force"],
                cwd=dashboard_path,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                st.success("Data certification complete! Loading dashboard...")
                st.cache_data.clear()
                time.sleep(1)
                st.rerun()
            else:
                st.error(f"Pipeline failed: {result.stderr}")
                st.markdown("""
                <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid #ef4444; border-radius: 12px; padding: 20px; margin: 20px 0;">
                    <p style="color: #a0aec0;">Please run manually from terminal:</p>
                    <code style="color: #10b981;">python pipelines/run_pipeline.py --force</code>
                </div>
                """, unsafe_allow_html=True)
        return
    
    # =================================================================
    # LOAD CERTIFIED DATA (Never raw CSVs)
    # =================================================================
    
    loading_placeholder = st.empty()
    
    try:
        with loading_placeholder.container():
            st.markdown("""
            <div style="background: rgba(26, 41, 66, 0.9); padding: 40px; border-radius: 12px; text-align: center; border: 1px solid rgba(212, 175, 55, 0.3);">
                <div style="font-size: 48px; margin-bottom: 20px;">🏛️</div>
                <h2 style="color: #ffffff; margin-bottom: 10px;">UIDAI Governance Intelligence</h2>
                <p style="color: #d4af37; margin-bottom: 20px;">Loading Certified Data...</p>
            </div>
            """, unsafe_allow_html=True)
            progress_bar = st.progress(0, text="Loading certified data pipeline...")
        
        # Load CERTIFIED data (not raw CSVs)
        progress_bar.progress(20, text="Loading certified enrollment data...")
        df_enrol, df_demo, df_bio, df_unified = load_certified_data()
        
        progress_bar.progress(60, text="Certified data loaded with pre-computed indices...")
        # Risk indices already computed by pipeline
        risk_df = calculate_all_risks(df_enrol, df_demo, df_bio, df_unified)
        
        progress_bar.progress(90, text="Preparing visualizations...")
        time.sleep(0.3)  # Brief pause for UX
        
        progress_bar.progress(100, text="Command Center Ready!")
        time.sleep(0.2)
        
        # Clear loading UI
        loading_placeholder.empty()
        
    except Exception as e:
        loading_placeholder.empty()
        st.markdown(f"""
        <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid #ef4444; border-radius: 12px; padding: 30px; margin: 20px 0;">
            <h2 style="color: #ef4444; margin-top: 0;">Data Loading Error</h2>
            <p style="color: #ffffff;">Could not load certified data files.</p>
            <p style="color: #a0aec0;">Please run the certification pipeline:</p>
            <code style="color: #10b981;">python pipelines/run_pipeline.py --force</code>
            <p style="color: #f59e0b; margin-top: 15px;"><strong>Error:</strong> {str(e)}</p>
        </div>
        """, unsafe_allow_html=True)
        import traceback
        with st.expander("Technical Details"):
            st.code(traceback.format_exc())
        return
    
    # Check if we have data
    if len(risk_df) == 0:
        st.warning("No data loaded. Please check that the data folders contain CSV files.")
        return
    
    # Calculate stats for KPIs
    stats = {
        'total_enrollments': int(df_enrol['total_enrollment'].sum()) if 'total_enrollment' in df_enrol.columns else 0,
        'total_demographic_updates': int(df_demo['total_demo'].sum()) if 'total_demo' in df_demo.columns else 0,
        'total_biometric_updates': int(df_bio['total_biometric'].sum()) if 'total_biometric' in df_bio.columns else 0,
        'states_covered': risk_df['state'].nunique() if 'state' in risk_df.columns else 0,
        'districts_covered': len(risk_df),
        'critical_districts': len(risk_df[risk_df['AESI'] > 75]) if 'AESI' in risk_df.columns else 0,
        'high_risk_districts': len(risk_df[risk_df['AESI'] > 50]) if 'AESI' in risk_df.columns else 0,
    }
    
    # Store data in session state for cross-dashboard access
    st.session_state['risk_df'] = risk_df
    st.session_state['df_enrol'] = df_enrol
    st.session_state['df_demo'] = df_demo
    st.session_state['df_bio'] = df_bio
    st.session_state['stats'] = stats
    
    
    # Populate states list for Compare Mode
    if 'state' in risk_df.columns:
        st.session_state['all_states'] = sorted(risk_df['state'].unique().tolist())
    
    
    # =================================================================
    
    # =================================================================
    # COMPARE MODE - Set flag to show dedicated comparison page
    # =================================================================
    show_compare_page = False
    if st.session_state.get('compare_mode', False):
        compare_selections = st.session_state.get('compare_selections')
        if compare_selections and len(compare_selections) == 2:
            state1, state2 = compare_selections
            if state1 != state2 and state1 != 'Loading...' and state2 != 'Loading...':
                show_compare_page = True
    
    # =================================================================
    # EXPORT FUNCTIONALITY
    # =================================================================
    if st.session_state.get('export_mode', False):
        st.markdown("""
        <div style="background: linear-gradient(90deg, rgba(16, 185, 129, 0.2) 0%, rgba(26, 41, 66, 0.8) 100%);
                    padding: 20px; border-radius: 12px; margin-bottom: 20px; border: 1px solid #10b981;">
            <h3 style="color: #10b981; margin: 0 0 15px 0;">📥 Export Dashboard Data</h3>
            <p style="color: #a0aec0; margin-bottom: 15px;">Download data as CSV files for offline analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.download_button(
                label="📊 Risk Analysis",
                data=risk_df.to_csv(index=False),
                file_name="uidai_risk_analysis.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            st.download_button(
                label="📝 Enrollment Data",
                data=df_enrol.to_csv(index=False),
                file_name="uidai_enrollment.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col3:
            st.download_button(
                label="👤 Demographic Data",
                data=df_demo.to_csv(index=False),
                file_name="uidai_demographic.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col4:
            st.download_button(
                label="🔐 Biometric Data",
                data=df_bio.to_csv(index=False),
                file_name="uidai_biometric.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        col_close = st.columns([1, 2, 1])[1]
        with col_close:
            if st.button("✖ CLOSE EXPORT PANEL", use_container_width=True, type="primary"):
                del st.session_state['export_mode']
                st.rerun()
        
        st.markdown("---")
    
    # Render header banner
    render_header(stats)
    
    # =================================================================
    # DEDICATED COMPARISON PAGE
    # =================================================================
    if show_compare_page:
        state1, state2 = st.session_state.get('compare_selections', ('', ''))
        
        # Header with Back to Home button
        col_back, col_title = st.columns([1, 5])
        with col_back:
            if st.button("Back to Home", use_container_width=True, type="primary"):
                st.session_state['compare_mode'] = False
                st.rerun()
        
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, rgba(139, 92, 246, 0.2) 0%, rgba(59, 130, 246, 0.2) 100%);
                    padding: 20px; border-radius: 12px; margin: 15px 0 25px 0; border: 1px solid #8b5cf6;">
            <h2 style="color: #8b5cf6; margin: 0;">State Comparison</h2>
            <p style="color: #a0aec0; margin: 5px 0 0 0;">{state1} vs {state2}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Get data for both states
        state1_data = risk_df[risk_df['state'] == state1] if 'state' in risk_df.columns else pd.DataFrame()
        state2_data = risk_df[risk_df['state'] == state2] if 'state' in risk_df.columns else pd.DataFrame()
        
        # Side-by-side metrics
        col1, col_vs, col2 = st.columns([3, 1, 3])
        
        with col1:
            st.markdown(f"###{state1}")
            if len(state1_data) > 0:
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Districts", len(state1_data))
                    st.metric("Avg AESI", f"{state1_data['AESI'].mean():.1f}" if 'AESI' in state1_data.columns else "N/A")
                with m2:
                    st.metric("Critical", len(state1_data[state1_data['AESI'] > 75]) if 'AESI' in state1_data.columns else 0)
                    st.metric("High Risk", len(state1_data[state1_data['AESI'] > 50]) if 'AESI' in state1_data.columns else 0)
            else:
                st.warning("No data available")
        
        with col_vs:
            st.markdown("<div style='text-align: center; padding-top: 50px;'><h1 style='color: #8b5cf6;'>VS</h1></div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"###{state2}")
            if len(state2_data) > 0:
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Districts", len(state2_data))
                    st.metric("Avg AESI", f"{state2_data['AESI'].mean():.1f}" if 'AESI' in state2_data.columns else "N/A")
                with m2:
                    st.metric("Critical", len(state2_data[state2_data['AESI'] > 75]) if 'AESI' in state2_data.columns else 0)
                    st.metric("High Risk", len(state2_data[state2_data['AESI'] > 50]) if 'AESI' in state2_data.columns else 0)
            else:
                st.warning("No data available")
        
        st.markdown("---")
        
        # Comparison bar chart
        if len(state1_data) > 0 and len(state2_data) > 0:
            import plotly.graph_objects as go
            
            metrics = ['Districts', 'Avg AESI', 'Critical Districts', 'High Risk Districts']
            state1_vals = [
                len(state1_data),
                state1_data['AESI'].mean() if 'AESI' in state1_data.columns else 0,
                len(state1_data[state1_data['AESI'] > 75]) if 'AESI' in state1_data.columns else 0,
                len(state1_data[state1_data['AESI'] > 50]) if 'AESI' in state1_data.columns else 0
            ]
            state2_vals = [
                len(state2_data),
                state2_data['AESI'].mean() if 'AESI' in state2_data.columns else 0,
                len(state2_data[state2_data['AESI'] > 75]) if 'AESI' in state2_data.columns else 0,
                len(state2_data[state2_data['AESI'] > 50]) if 'AESI' in state2_data.columns else 0
            ]
            
            fig = go.Figure(data=[
                go.Bar(name=state1, x=metrics, y=state1_vals, marker_color='#8b5cf6'),
                go.Bar(name=state2, x=metrics, y=state2_vals, marker_color='#3b82f6')
            ])
            fig.update_layout(
                barmode='group',
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(26, 41, 66, 0.5)',
                font=dict(color='white'),
                legend=dict(font=dict(color='white'))
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # District tables
        st.markdown("### 📋 District Details")
        tab1, tab2 = st.tabs([f"{state1} Districts", f"{state2} Districts"])
        
        with tab1:
            if len(state1_data) > 0:
                display_cols = ['district', 'AESI'] + [c for c in ['total_enrollment', 'total_activity'] if c in state1_data.columns]
                st.dataframe(state1_data[display_cols].sort_values('AESI', ascending=False) if 'AESI' in state1_data.columns else state1_data, use_container_width=True)
        
        with tab2:
            if len(state2_data) > 0:
                display_cols = ['district', 'AESI'] + [c for c in ['total_enrollment', 'total_activity'] if c in state2_data.columns]
                st.dataframe(state2_data[display_cols].sort_values('AESI', ascending=False) if 'AESI' in state2_data.columns else state2_data, use_container_width=True)
    
    # Render selected dashboard (only if not in compare page)
    elif "National" in selected_dashboard:
        render_national_dashboard(stats, risk_df, df_enrol=df_enrol, df_demo=df_demo, df_bio=df_bio)
        
    elif "Enrolment" in selected_dashboard or "Enrollment" in selected_dashboard:
        render_enrolment_dashboard(df_enrol, risk_df)
        
    elif "Biometric" in selected_dashboard:
        render_biometric_dashboard(df_bio, risk_df)
        
    elif "Demographic" in selected_dashboard:
        render_demographic_dashboard(df_demo, risk_df)
        
    elif "Policy" in selected_dashboard:
        render_policy_dashboard(risk_df, df_enrol, df_bio, stats)
    
    # Auto-refresh mechanism
    if auto_refresh:
        # Use session state to track time
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = time.time()
        
        elapsed = time.time() - st.session_state.last_refresh
        
        # Refresh every 5 minutes (300 seconds)
        if elapsed > 300:
            st.session_state.last_refresh = time.time()
            st.cache_data.clear()
            st.rerun()


def render_policy_dashboard(risk_df, df_enrol, df_bio, stats):
    """
    ================================================================================
    GOVERNANCE RECOMMENDATION & POLICY IMPLEMENTATION ENGINE
    ================================================================================
    Pure intelligence content only. System controls are in the global sidebar.
    ================================================================================
    """
    from datetime import datetime
    import plotly.graph_objects as go
    
    # =============================================================================
    # CALCULATE METRICS (used throughout the page)
    # =============================================================================
    
    critical_count = len(risk_df[risk_df['AESI'] > 75]) if 'AESI' in risk_df.columns else 0
    high_count = len(risk_df[(risk_df['AESI'] > 50) & (risk_df['AESI'] <= 75)]) if 'AESI' in risk_df.columns else 0
    moderate_count = len(risk_df[(risk_df['AESI'] > 25) & (risk_df['AESI'] <= 50)]) if 'AESI' in risk_df.columns else 0
    low_count = len(risk_df[risk_df['AESI'] <= 25]) if 'AESI' in risk_df.columns else 0
    
    total_at_risk = critical_count + high_count
    total_activity_critical = risk_df[risk_df['AESI'] > 75]['total_activity'].sum() if 'AESI' in risk_df.columns and 'total_activity' in risk_df.columns else 0
    total_activity_high = risk_df[(risk_df['AESI'] > 50) & (risk_df['AESI'] <= 75)]['total_activity'].sum() if 'AESI' in risk_df.columns else 0
    estimated_population_exposed = int((total_activity_critical + total_activity_high) * 1.5)
    estimated_leakage = int(total_activity_critical * 500 * 0.02)
    
    # Determine risk status for styling
    if critical_count > 20:
        risk_status = "CRITICAL"
        status_color = "#dc2626"
    elif critical_count > 10 or high_count > 50:
        risk_status = "ELEVATED"
        status_color = "#f59e0b"
    elif critical_count > 5 or high_count > 25:
        risk_status = "HEIGHTENED"
        status_color = "#eab308"
    else:
        risk_status = "NORMAL"
        status_color = "#10b981"
    
    # =============================================================================
    # SECTION 1: POLICY INTELLIGENCE HEADER (Domain-specific, not system-level)
    # =============================================================================
    
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, rgba(26, 41, 65, 0.95) 0%, rgba(12, 25, 41, 0.95) 100%);
                padding: 16px 20px; border-radius: 4px; margin-bottom: 16px; border-bottom: 2px solid {status_color};">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="color: {status_color}; font-size: 10px; font-weight: 600; letter-spacing: 1px;">POLICY INTELLIGENCE</div>
                <div style="color: #ffffff; font-size: 18px; font-weight: 700;">Governance Recommendation Engine</div>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="width: 10px; height: 10px; background: {status_color}; border-radius: 50%;"></span>
                <span style="color: {status_color}; font-size: 11px; font-weight: 600;">RISK STATUS: {risk_status}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Stats Row (matching enrollment section UI style)
    col1, col2, col3, col4 = st.columns(4)
    
    kpi_data = [
        ("CRITICAL DISTRICTS", str(critical_count), "#dc2626"),
        ("HIGH RISK", str(high_count), "#f59e0b"),
        ("ESTIMATED LEAKAGE", f"₹{estimated_leakage/10000000:.1f} Cr", "#ef4444"),
        ("POPULATION EXPOSED", f"{estimated_population_exposed:,}", "#ffffff")
    ]
    
    for col, (label, value, color) in zip([col1, col2, col3, col4], kpi_data):
        with col:
            if value and value != "0":
                st.markdown(f"""
                <div style="background: rgba(26, 41, 65, 0.6); padding: 12px; border-radius: 4px; text-align: center; border: 1px solid rgba(212, 175, 55, 0.1);">
                    <div style="color: #a0aec0; font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px;">{label}</div>
                    <div style="color: {color}; font-size: 22px; font-weight: 700; margin-top: 4px;">{value}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: rgba(26, 41, 65, 0.6); padding: 12px; border-radius: 4px; text-align: center;">
                    <div style="color: #a0aec0; font-size: 10px; text-transform: uppercase;">{label}</div>
                    <div style="color: #f59e0b; font-size: 12px; margin-top: 4px;">Data Not Available</div>
                </div>
                """, unsafe_allow_html=True)
    
    # =============================================================================
    # SECTION 2: IMMEDIATE ACTION REQUIRED — TOP 10 COMMAND TABLE
    # =============================================================================
    
    st.markdown("""
    <div style="background: linear-gradient(90deg, rgba(220, 38, 38, 0.2) 0%, rgba(26, 41, 65, 0.9) 100%);
                padding: 16px 20px; border-radius: 4px; margin-bottom: 16px; border-left: 4px solid #dc2626;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="color: #dc2626; font-size: 11px; font-weight: 700; letter-spacing: 2px;">▌ IMMEDIATE ACTION REQUIRED</div>
                <div style="color: #ffffff; font-size: 18px; font-weight: 700;">Top 10 Priority Districts — Operational Task List</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if 'AESI' in risk_df.columns:
        priority_df = risk_df.nlargest(10, 'AESI').copy()
        priority_df['Rank'] = range(1, len(priority_df) + 1)
        
        # Determine risk drivers
        def get_risk_drivers(row):
            drivers = []
            if row.get('EPI_normalized', 0) > 60: drivers.append("High Enrollment Pressure")
            if row.get('DULI_normalized', 0) > 60: drivers.append("Demo Update Load")
            if row.get('BRS_normalized', 0) > 60: drivers.append("Biometric Stress")
            if row.get('security_ratio', 1) < 0.5: drivers.append("Low Security Compliance")
            if row.get('migration_intensity', 0) > 5: drivers.append("Migration Anomaly")
            return ", ".join(drivers[:2]) if drivers else "Multi-factor stress"
        
        # Determine recommended actions
        def get_action(row):
            aesi = row.get('AESI', 0)
            if aesi >= 85: return "Emergency audit + operator suspension"
            elif aesi >= 75: return "Deploy mobile units + fraud team"
            elif aesi >= 65: return "Capacity review + staff augmentation"
            else: return "Enhanced monitoring"
        
        # Determine priority level
        def get_priority(aesi):
            if aesi >= 85: return "P0 - IMMEDIATE"
            elif aesi >= 75: return "P1 - URGENT"
            elif aesi >= 65: return "P2 - HIGH"
            else: return "P3 - ELEVATED"
        
        priority_df['Risk_Drivers'] = priority_df.apply(get_risk_drivers, axis=1)
        priority_df['Recommended_Action'] = priority_df.apply(get_action, axis=1)
        priority_df['Priority'] = priority_df['AESI'].apply(get_priority)
        
        display_df = priority_df[['Rank', 'state', 'district', 'AESI', 'total_activity', 'Risk_Drivers', 'Recommended_Action', 'Priority']].copy()
        display_df.columns = ['Rank', 'State', 'District', 'Risk Score', 'Activity Volume', 'Risk Drivers', 'Recommended Actions', 'Priority']
        display_df['Risk Score'] = display_df['Risk Score'].round(1)
        display_df['Activity Volume'] = display_df['Activity Volume'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A")
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Rank': st.column_config.NumberColumn('Rank', width='small'),
                'Risk Score': st.column_config.ProgressColumn('Risk Score', format='%.1f', min_value=0, max_value=100),
                'Priority': st.column_config.TextColumn('Priority', width='medium')
            }
        )
    
    st.markdown("---")
    
    # =============================================================================
    # SECTION 3: STRATEGIC SUMMARY PANEL — NATIONAL SYNTHESIS
    # =============================================================================
    
    st.markdown("""
    <div style="background: linear-gradient(90deg, rgba(59, 130, 246, 0.15) 0%, rgba(26, 41, 65, 0.9) 100%);
                padding: 16px 20px; border-radius: 4px; margin-bottom: 16px; border-left: 4px solid #3b82f6;">
        <div style="color: #3b82f6; font-size: 11px; font-weight: 700; letter-spacing: 2px;">▌ STRATEGIC SUMMARY</div>
        <div style="color: #ffffff; font-size: 16px; font-weight: 600;">National Synthesis — Connecting District Risks to National Consequences</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Calculate strategic metrics
    total_activity = risk_df['total_activity'].sum() if 'total_activity' in risk_df.columns else 0
    at_risk_activity = (total_activity_critical + total_activity_high)
    at_risk_percentage = (at_risk_activity / total_activity * 100) if total_activity > 0 else 0
    annual_leakage_exposure = estimated_leakage * 12  # Annualized
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="background: rgba(26, 41, 65, 0.8); padding: 20px; border-radius: 8px; border: 1px solid rgba(212, 175, 55, 0.2); height: 100%;">
            <h4 style="color: #d4af37; margin: 0 0 15px 0; font-size: 14px;">📊 IMPACT ASSESSMENT</h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                <div>
                    <div style="color: #a0aec0; font-size: 10px; text-transform: uppercase;">Total At-Risk Population</div>
                    <div style="color: #ffffff; font-size: 22px; font-weight: 700;">{estimated_population_exposed:,}</div>
                    <div style="color: #f59e0b; font-size: 11px;">({at_risk_percentage:.1f}% of total activity)</div>
                </div>
                <div>
                    <div style="color: #a0aec0; font-size: 10px; text-transform: uppercase;">Annual Leakage Exposure</div>
                    <div style="color: #ef4444; font-size: 22px; font-weight: 700;">₹{annual_leakage_exposure/10000000:.1f} Cr</div>
                    <div style="color: #a0aec0; font-size: 11px;">Fraud + operational loss</div>
                </div>
                <div>
                    <div style="color: #a0aec0; font-size: 10px; text-transform: uppercase;">Districts Requiring Action</div>
                    <div style="color: #ffffff; font-size: 22px; font-weight: 700;">{total_at_risk}</div>
                    <div style="color: #dc2626; font-size: 11px;">{critical_count} Critical + {high_count} High</div>
                </div>
                <div>
                    <div style="color: #a0aec0; font-size: 10px; text-transform: uppercase;">States Affected</div>
                    <div style="color: #ffffff; font-size: 22px; font-weight: 700;">{risk_df[risk_df['AESI'] > 50]['state'].nunique() if 'AESI' in risk_df.columns else 0}</div>
                    <div style="color: #a0aec0; font-size: 11px;">With high-risk districts</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: rgba(26, 41, 65, 0.8); padding: 20px; border-radius: 8px; border: 1px solid rgba(212, 175, 55, 0.2); height: 100%;">
            <h4 style="color: #d4af37; margin: 0 0 15px 0; font-size: 14px;">🎯 ACTIVATED POLICY LEVERS</h4>
            <div style="display: flex; flex-direction: column; gap: 10px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="color: {'#10b981' if critical_count > 5 else '#4b5563'};">{'●' if critical_count > 5 else '○'}</span>
                    <span style="color: #ffffff; font-size: 12px;"><strong>POL-001:</strong> Emergency Audit Protocol</span>
                    <span style="color: {'#10b981' if critical_count > 5 else '#6b7280'}; font-size: 10px;">{'ACTIVE' if critical_count > 5 else 'STANDBY'}</span>
                </div>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="color: {'#10b981' if high_count > 20 else '#4b5563'};">{'●' if high_count > 20 else '○'}</span>
                    <span style="color: #ffffff; font-size: 12px;"><strong>POL-002:</strong> Mobile Unit Rapid Deployment</span>
                    <span style="color: {'#10b981' if high_count > 20 else '#6b7280'}; font-size: 10px;">{'ACTIVE' if high_count > 20 else 'STANDBY'}</span>
                </div>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="color: {'#10b981' if estimated_leakage > 10000000 else '#4b5563'};">{'●' if estimated_leakage > 10000000 else '○'}</span>
                    <span style="color: #ffffff; font-size: 12px;"><strong>POL-003:</strong> PDS/DBT Cross-Verification</span>
                    <span style="color: {'#10b981' if estimated_leakage > 10000000 else '#6b7280'}; font-size: 10px;">{'ACTIVE' if estimated_leakage > 10000000 else 'STANDBY'}</span>
                </div>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="color: {'#10b981' if critical_count > 10 else '#4b5563'};">{'●' if critical_count > 10 else '○'}</span>
                    <span style="color: #ffffff; font-size: 12px;"><strong>POL-004:</strong> Joint State-UIDAI Task Force</span>
                    <span style="color: {'#10b981' if critical_count > 10 else '#6b7280'}; font-size: 10px;">{'ACTIVE' if critical_count > 10 else 'STANDBY'}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # =============================================================================
    # SECTION 4: 72-HOUR ACTIONS (CRISIS MODE DIRECTIVES)
    # =============================================================================
    
    st.markdown("""
    <div style="background: linear-gradient(90deg, rgba(220, 38, 38, 0.2) 0%, rgba(245, 158, 11, 0.1) 100%);
                padding: 16px 20px; border-radius: 4px; margin-bottom: 16px; border-left: 4px solid #f59e0b;">
        <div style="color: #f59e0b; font-size: 11px; font-weight: 700; letter-spacing: 2px;">▌ RECOMMENDED 72-HOUR ACTIONS</div>
        <div style="color: #ffffff; font-size: 16px; font-weight: 600;">Emergency Operations Directive — Crisis Response Protocol</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Get top 5 critical districts for specific actions
    if 'AESI' in risk_df.columns:
        top5_critical = risk_df.nlargest(5, 'AESI')[['state', 'district', 'AESI']].values.tolist()
        top5_names = ", ".join([f"{d[1]} ({d[0]})" for d in top5_critical[:3]])
        
        # Identify state clusters
        critical_by_state = risk_df[risk_df['AESI'] > 75].groupby('state').size().sort_values(ascending=False)
        top_cluster_states = critical_by_state.head(2).index.tolist() if len(critical_by_state) > 0 else ["Bihar", "UP"]
    else:
        top5_names = "Top critical districts"
        top_cluster_states = ["Bihar", "UP"]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="background: rgba(26, 41, 65, 0.8); padding: 20px; border-radius: 8px; border: 1px solid rgba(239, 68, 68, 0.3);">
            <h4 style="color: #ef4444; margin: 0 0 15px 0; font-size: 13px;">🚨 IMMEDIATE (0-24 Hours)</h4>
            <div style="color: #ffffff; font-size: 12px; line-height: 1.8;">
                <div style="display: flex; align-items: flex-start; gap: 10px; margin-bottom: 8px;">
                    <input type="checkbox" style="margin-top: 4px;"> 
                    <span>Suspend non-essential demographic updates in <strong style="color: #f59e0b;">{top5_names}</strong></span>
                </div>
                <div style="display: flex; align-items: flex-start; gap: 10px; margin-bottom: 8px;">
                    <input type="checkbox" style="margin-top: 4px;"> 
                    <span>Issue operator compliance notice to all centers in critical districts</span>
                </div>
                <div style="display: flex; align-items: flex-start; gap: 10px; margin-bottom: 8px;">
                    <input type="checkbox" style="margin-top: 4px;"> 
                    <span>Activate fraud detection batch processing for flagged transactions</span>
                </div>
                <div style="display: flex; align-items: flex-start; gap: 10px;">
                    <input type="checkbox" style="margin-top: 4px;"> 
                    <span>Notify RGI Office and state nodal officers</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: rgba(26, 41, 65, 0.8); padding: 20px; border-radius: 8px; border: 1px solid rgba(245, 158, 11, 0.3);">
            <h4 style="color: #f59e0b; margin: 0 0 15px 0; font-size: 13px;">⚠️ SHORT-TERM (24-72 Hours)</h4>
            <div style="color: #ffffff; font-size: 12px; line-height: 1.8;">
                <div style="display: flex; align-items: flex-start; gap: 10px; margin-bottom: 8px;">
                    <input type="checkbox" style="margin-top: 4px;"> 
                    <span>Deploy joint UIDAI + state audit teams to <strong style="color: #f59e0b;">{', '.join(top_cluster_states)}</strong> cluster</span>
                </div>
                <div style="display: flex; align-items: flex-start; gap: 10px; margin-bottom: 8px;">
                    <input type="checkbox" style="margin-top: 4px;"> 
                    <span>Trigger PDS/DBT cross-verification batch for flagged Aadhaar numbers</span>
                </div>
                <div style="display: flex; align-items: flex-start; gap: 10px; margin-bottom: 8px;">
                    <input type="checkbox" style="margin-top: 4px;"> 
                    <span>Escalate top critical district to Command Center review</span>
                </div>
                <div style="display: flex; align-items: flex-start; gap: 10px;">
                    <input type="checkbox" style="margin-top: 4px;"> 
                    <span>Prepare emergency mobile enrollment deployment plan</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # =============================================================================
    # SECTION 5: LONGER-TERM MITIGATION ROADMAP (6-12 MONTHS)
    # =============================================================================
    
    st.markdown("""
    <div style="background: linear-gradient(90deg, rgba(139, 92, 246, 0.15) 0%, rgba(26, 41, 65, 0.9) 100%);
                padding: 16px 20px; border-radius: 4px; margin-bottom: 16px; border-left: 4px solid #8b5cf6;">
        <div style="color: #8b5cf6; font-size: 11px; font-weight: 700; letter-spacing: 2px;">▌ LONGER-TERM MITIGATION ROADMAP</div>
        <div style="color: #ffffff; font-size: 16px; font-weight: 600;">Strategic Planning Horizon — 6-12 Months</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: rgba(26, 41, 65, 0.8); padding: 20px; border-radius: 8px; border: 1px solid rgba(139, 92, 246, 0.3); height: 100%;">
            <div style="color: #8b5cf6; font-size: 24px; margin-bottom: 10px;"></div>
            <h4 style="color: #ffffff; margin: 0 0 10px 0; font-size: 14px;">Infrastructure Scaling</h4>
            <ul style="color: #a0aec0; font-size: 11px; padding-left: 15px; margin: 0;">
                <li style="margin-bottom: 6px;">New permanent centers in UP, Bihar, Maharashtra (50+ each)</li>
                <li style="margin-bottom: 6px;">Upgrade bandwidth in Northeast region</li>
                <li style="margin-bottom: 6px;">Server capacity expansion for peak load</li>
                <li>Backup power infrastructure in rural areas</li>
            </ul>
            <div style="color: #8b5cf6; font-size: 10px; margin-top: 12px;">Est. Investment: ₹200-250 Cr</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: rgba(26, 41, 65, 0.8); padding: 20px; border-radius: 8px; border: 1px solid rgba(139, 92, 246, 0.3); height: 100%;">
            <div style="color: #10b981; font-size: 24px; margin-bottom: 10px;"></div>
            <h4 style="color: #ffffff; margin: 0 0 10px 0; font-size: 14px;">Capacity Building Programs</h4>
            <ul style="color: #a0aec0; font-size: 11px; padding-left: 15px; margin: 0;">
                <li style="margin-bottom: 6px;">School-based biometric update camps (MBU preparation)</li>
                <li style="margin-bottom: 6px;">Operator certification and retraining program</li>
                <li style="margin-bottom: 6px;">State-level training academies</li>
                <li>Fraud detection skill development</li>
            </ul>
            <div style="color: #10b981; font-size: 10px; margin-top: 12px;">Est. Investment: ₹50-75 Cr</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: rgba(26, 41, 65, 0.8); padding: 20px; border-radius: 8px; border: 1px solid rgba(139, 92, 246, 0.3); height: 100%;">
            <div style="color: #f59e0b; font-size: 24px; margin-bottom: 10px;"></div>
            <h4 style="color: #ffffff; margin: 0 0 10px 0; font-size: 14px;">Mobile & Outreach Expansion</h4>
            <ul style="color: #a0aec0; font-size: 11px; padding-left: 15px; margin: 0;">
                <li style="margin-bottom: 6px;">Mobile enrollment vans in Northeast & tribal areas</li>
                <li style="margin-bottom: 6px;">Doorstep service for senior citizens</li>
                <li style="margin-bottom: 6px;">Integration with CSC network expansion</li>
                <li>Aadhaar Seva Kendras in underserved blocks</li>
            </ul>
            <div style="color: #f59e0b; font-size: 10px; margin-top: 12px;">Est. Investment: ₹100-150 Cr</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # =============================================================================
    # SECTION 6: COST-BENEFIT & SCENARIO ANALYSIS (Collapsible)
    # =============================================================================
    
    with st.expander("Investment ROI Analysis & Scenario Planning", expanded=False):
        tab1, tab2 = st.tabs(["Cost-Benefit Analysis", "Scenario Planning"])
        
        with tab1:
            interventions = ['New Centers', 'Mobile Units', 'Staff Training', 'Fraud Teams', 'Awareness', 'Tech Upgrade']
            costs = [15, 25, 2, 10, 3, 20]
            benefits = [25, 40, 5, 50, 8, 35]
            roi = [(b/c)*100 - 100 for b, c in zip(benefits, costs)]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Cost (₹ Lakhs/unit)', x=interventions, y=costs, marker_color='#ef4444'))
            fig.add_trace(go.Bar(name='Benefit (₹ Lakhs/unit)', x=interventions, y=benefits, marker_color='#10b981'))
            
            fig.update_layout(
                barmode='group',
                title=dict(text='Intervention Cost vs Expected Benefit', font=dict(color='#d4af37')),
                plot_bgcolor='rgba(26, 41, 66, 0.5)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=400,
                xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.1)', title='₹ Lakhs')
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # ROI Summary
            st.markdown("**Highest ROI Interventions:**")
            roi_df = pd.DataFrame({'Intervention': interventions, 'ROI %': roi}).sort_values('ROI %', ascending=False)
            st.dataframe(roi_df, use_container_width=True, hide_index=True)
        
        with tab2:
            scenario = st.selectbox(
                "Select Scenario",
                ["Northeast Expansion", "Bihar-Jharkhand Focus", "Pan-India Coverage", "High-Risk Priority"]
            )
            
            scenario_data = {
                "Northeast Expansion": {"coverage": "+25-30%", "risk_reduction": "-15-20%", "investment": "₹180-220 Cr", "payback": "2-3 years"},
                "Bihar-Jharkhand Focus": {"coverage": "+18-22%", "risk_reduction": "-25-30%", "investment": "₹120-150 Cr", "payback": "1.5-2 years"},
                "Pan-India Coverage": {"coverage": "+12-15%", "risk_reduction": "-10-12%", "investment": "₹400-500 Cr", "payback": "3-4 years"},
                "High-Risk Priority": {"coverage": "+8-10%", "risk_reduction": "-35-40%", "investment": "₹80-100 Cr", "payback": "1-1.5 years"}
            }
            
            data = scenario_data[scenario]
            
            st.markdown(f"""
            <div style="background: rgba(26, 41, 66, 0.8); padding: 20px; border-radius: 8px; border: 1px solid rgba(212, 175, 55, 0.3);">
                <h4 style="color: #d4af37; margin: 0 0 15px 0;">{scenario} Scenario</h4>
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
                    <div>
                        <div style="color: #a0aec0; font-size: 10px;">Coverage Increase</div>
                        <div style="color: #10b981; font-size: 18px; font-weight: 700;">{data['coverage']}</div>
                    </div>
                    <div>
                        <div style="color: #a0aec0; font-size: 10px;">Risk Reduction</div>
                        <div style="color: #10b981; font-size: 18px; font-weight: 700;">{data['risk_reduction']}</div>
                    </div>
                    <div>
                        <div style="color: #a0aec0; font-size: 10px;">Investment Required</div>
                        <div style="color: #f59e0b; font-size: 18px; font-weight: 700;">{data['investment']}</div>
                    </div>
                    <div>
                        <div style="color: #a0aec0; font-size: 10px;">Payback Period</div>
                        <div style="color: #ffffff; font-size: 18px; font-weight: 700;">{data['payback']}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
