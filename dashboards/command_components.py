"""
================================================================================
UIDAI COMMAND CENTER COMPONENTS
================================================================================
Shared components for Government Command-and-Control Dashboard transformation.

Components:
- Situation Intelligence Panel (replaces text explanations)
- Priority Action Table (Top 10 districts requiring action)
- Collapsible Methodology Section
- Protected KPI Display (handles missing data)
- Command Header styling
================================================================================
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Optional, Union


# =============================================================================
# INSTITUTIONAL COLOR SCHEME - NIC/UIDAI Standard
# =============================================================================
COMMAND_COLORS = {
    'bg_primary': '#0c1929',        # Dark navy background
    'bg_secondary': '#1a2941',      # Lighter navy
    'accent_gold': '#d4af37',       # UIDAI gold accent
    'text_primary': '#ffffff',      # White text
    'text_secondary': '#a0aec0',    # Muted text
    'critical': '#dc2626',          # Red - Critical
    'warning': '#f59e0b',           # Amber - Warning
    'caution': '#eab308',           # Yellow - Caution
    'safe': '#10b981',              # Green - Safe
    'info': '#3b82f6',              # Blue - Info
}


# =============================================================================
# SITUATION INTELLIGENCE PANEL
# =============================================================================
def render_situation_panel(
    title: str,
    alerts: List[Dict[str, Union[str, int]]],
    severity: str = "info"
):
    """
    Render a Situation Intelligence Panel that replaces verbose text explanations.
    
    Args:
        title: Panel header (e.g., "CURRENT SITUATION")
        alerts: List of dicts with 'count', 'message', and optional 'severity'
        severity: Default severity level ('critical', 'warning', 'info', 'safe')
    """
    severity_styles = {
        'critical': {'border': '#dc2626', 'bg': 'rgba(220, 38, 38, 0.1)', 'icon': '🔴'},
        'warning': {'border': '#f59e0b', 'bg': 'rgba(245, 158, 11, 0.1)', 'icon': '🟡'},
        'info': {'border': '#3b82f6', 'bg': 'rgba(59, 130, 246, 0.1)', 'icon': '🔵'},
        'safe': {'border': '#10b981', 'bg': 'rgba(16, 185, 129, 0.1)', 'icon': '🟢'},
    }
    
    style = severity_styles.get(severity, severity_styles['info'])
    
    alerts_html = ""
    for alert in alerts:
        alert_sev = alert.get('severity', severity)
        alert_style = severity_styles.get(alert_sev, style)
        count = alert.get('count', '')
        message = alert.get('message', '')
        
        alerts_html += f"""
        <div style="display: flex; align-items: center; gap: 12px; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
            <span style="font-size: 20px;">{alert_style['icon']}</span>
            <span style="color: {alert_style['border']}; font-size: 24px; font-weight: 700; min-width: 60px;">{count}</span>
            <span style="color: #e2e8f0; font-size: 14px;">{message}</span>
        </div>
        """
    
    st.markdown(f"""
    <div style="background: {style['bg']}; border-left: 4px solid {style['border']}; 
                padding: 16px; border-radius: 4px; margin-bottom: 16px;">
        <div style="color: {style['border']}; font-size: 11px; font-weight: 700; letter-spacing: 1px; margin-bottom: 12px;">
            ▌ {title.upper()}
        </div>
        {alerts_html}
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PRIORITY ACTION TABLE
# =============================================================================
def render_priority_action_table(
    df: pd.DataFrame,
    risk_column: str = 'AESI',
    title: str = "TOP 10 PRIORITY DISTRICTS",
    show_recommendations: bool = True
):
    """
    Render Priority Action Table showing top districts requiring immediate attention.
    
    Args:
        df: DataFrame with district-level risk data
        risk_column: Column name for risk score (AESI, BRS_normalized, DULI_normalized)
        title: Table title
        show_recommendations: Whether to show recommended action column
    """
    if df is None or len(df) == 0 or risk_column not in df.columns:
        st.warning("Priority data not available for current window")
        return
    
    # Generate priority table
    required_cols = ['state', 'district', risk_column]
    available_cols = [c for c in required_cols if c in df.columns]
    
    if len(available_cols) < 2:
        st.warning("Insufficient data for priority analysis")
        return
    
    priority_df = df.nlargest(10, risk_column)[available_cols].copy()
    priority_df['Rank'] = range(1, len(priority_df) + 1)
    
    # Calculate trend (simplified - based on risk level)
    def get_trend(score):
        if score >= 80: return "Rising"
        elif score >= 60: return "Increasing"
        elif score >= 40: return "Stable"
        else: return "Declining"
    
    priority_df['Trend'] = priority_df[risk_column].apply(get_trend)
    
    # Generate recommended action
    def get_action(score):
        if score >= 80: return "Immediate field intervention"
        elif score >= 60: return "Capacity augmentation needed"
        elif score >= 40: return "Enhanced monitoring"
        else: return "Continue standard operations"
    
    if show_recommendations:
        priority_df['Recommended Action'] = priority_df[risk_column].apply(get_action)
    
    # Reorder columns
    col_order = ['Rank', 'state', 'district', risk_column, 'Trend']
    if show_recommendations:
        col_order.append('Recommended Action')
    
    priority_df = priority_df[[c for c in col_order if c in priority_df.columns]]
    priority_df.columns = ['Rank', 'State', 'District', 'Risk Score', 'Trend'] + (['Action'] if show_recommendations else [])
    priority_df['Risk Score'] = priority_df['Risk Score'].round(1)
    
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, rgba(220, 38, 38, 0.15) 0%, rgba(26, 41, 65, 0.8) 100%);
                padding: 12px 16px; border-radius: 4px 4px 0 0; border-left: 4px solid #dc2626;">
        <span style="color: #dc2626; font-size: 12px; font-weight: 700; letter-spacing: 1px;">
            ▌ {title}
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    st.dataframe(
        priority_df, 
        use_container_width=True, 
        hide_index=True,
        column_config={
            'Risk Score': st.column_config.ProgressColumn(
                'Risk Score', format='%.1f', min_value=0, max_value=100
            )
        }
    )


# =============================================================================
# COLLAPSIBLE METHODOLOGY SECTION
# =============================================================================
def render_methodology_section(
    title: str,
    formula: str,
    components: List[Dict[str, str]],
    interpretation: str = None
):
    """
    Render collapsible methodology section for technical content.
    
    Args:
        title: Methodology section title
        formula: Mathematical formula or computation logic
        components: List of dicts with 'name' and 'description'
        interpretation: Optional interpretation guide
    """
    with st.expander(f"Methodology & Computation: {title}", expanded=False):
        st.markdown(f"""
        <div style="background: rgba(26, 41, 65, 0.5); padding: 16px; border-radius: 4px; border: 1px solid rgba(212, 175, 55, 0.2);">
            <div style="color: #d4af37; font-size: 12px; font-weight: 600; margin-bottom: 8px;">FORMULA</div>
            <code style="color: #e2e8f0; background: rgba(0,0,0,0.3); padding: 8px 12px; border-radius: 4px; display: block; font-size: 13px;">
                {formula}
            </code>
        </div>
        """, unsafe_allow_html=True)
        
        if components:
            st.markdown("""
            <div style="color: #d4af37; font-size: 12px; font-weight: 600; margin: 16px 0 8px 0;">COMPONENTS</div>
            """, unsafe_allow_html=True)
            
            for comp in components:
                st.markdown(f"""
                <div style="display: flex; gap: 8px; padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <span style="color: #3b82f6; font-weight: 600; min-width: 120px;">{comp['name']}</span>
                    <span style="color: #a0aec0;">{comp['description']}</span>
                </div>
                """, unsafe_allow_html=True)
        
        if interpretation:
            st.markdown(f"""
            <div style="margin-top: 16px; padding: 12px; background: rgba(59, 130, 246, 0.1); border-radius: 4px;">
                <div style="color: #3b82f6; font-size: 12px; font-weight: 600; margin-bottom: 6px;">INTERPRETATION</div>
                <div style="color: #e2e8f0; font-size: 13px;">{interpretation}</div>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# PROTECTED KPI DISPLAY
# =============================================================================
def render_protected_kpi(
    label: str,
    value: Union[int, float, str],
    delta: Optional[Union[int, float, str]] = None,
    show_unavailable: bool = True
):
    """
    Render KPI with protection against misleading zero values.
    
    Args:
        label: KPI label
        value: KPI value
        delta: Optional delta/change value
        show_unavailable: If True, shows "Data Not Available" for zero/None values
    """
    if value is None or (isinstance(value, (int, float)) and value == 0 and show_unavailable):
        st.markdown(f"""
        <div style="background: rgba(26, 41, 65, 0.5); padding: 12px; border-radius: 4px; text-align: center;">
            <div style="color: #a0aec0; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px;">{label}</div>
            <div style="color: #f59e0b; font-size: 14px; margin-top: 4px;">Data Not Available</div>
            <div style="color: #6b7280; font-size: 10px;">(Current Window)</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        if isinstance(value, (int, float)):
            if value >= 1e7:
                display_value = f"{value/1e7:.2f} Cr"
            elif value >= 1e5:
                display_value = f"{value/1e5:.2f} L"
            elif value >= 1e3:
                display_value = f"{value/1e3:.1f} K"
            else:
                display_value = f"{value:,.0f}" if isinstance(value, int) else f"{value:.1f}"
        else:
            display_value = str(value)
        
        delta_html = ""
        if delta is not None:
            delta_color = "#10b981" if (isinstance(delta, (int, float)) and delta >= 0) else "#ef4444"
            delta_html = f'<div style="color: {delta_color}; font-size: 12px; margin-top: 2px;">{delta}</div>'
        
        st.markdown(f"""
        <div style="background: rgba(26, 41, 65, 0.5); padding: 12px; border-radius: 4px; text-align: center; border: 1px solid rgba(212, 175, 55, 0.1);">
            <div style="color: #a0aec0; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px;">{label}</div>
            <div style="color: #ffffff; font-size: 24px; font-weight: 700; margin-top: 4px;">{display_value}</div>
            {delta_html}
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# COMMAND CENTER HEADER
# =============================================================================
def render_command_header(
    title: str,
    subtitle: str = None,
    status: str = "OPERATIONAL"
):
    """
    Render command center style header with status indicator.
    
    Args:
        title: Main header title
        subtitle: Optional subtitle
        status: Status indicator (OPERATIONAL, ALERT, CRITICAL)
    """
    status_colors = {
        'OPERATIONAL': '#10b981',
        'ALERT': '#f59e0b',
        'CRITICAL': '#dc2626'
    }
    status_color = status_colors.get(status.upper(), '#10b981')
    
    subtitle_html = f'<div style="color: #a0aec0; font-size: 13px; margin-top: 4px;">{subtitle}</div>' if subtitle else ''
    
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, rgba(26, 41, 65, 0.95) 0%, rgba(12, 25, 41, 0.95) 100%);
                padding: 20px; border-radius: 4px; margin-bottom: 20px; border-bottom: 2px solid #d4af37;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="color: #ffffff; font-size: 20px; font-weight: 700; letter-spacing: 0.5px;">{title}</div>
                {subtitle_html}
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="width: 10px; height: 10px; background: {status_color}; border-radius: 50%; animation: pulse 2s infinite;"></span>
                <span style="color: {status_color}; font-size: 11px; font-weight: 600; letter-spacing: 1px;">{status.upper()}</span>
            </div>
        </div>
    </div>
    <style>
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
        }}
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# MAP-FIRST SECTION HEADER
# =============================================================================
def render_map_section_header(domain: str, index_name: str, index_value: float = None):
    """
    Render header for map-first sections with domain indicator.
    
    Args:
        domain: Domain name (e.g., "ENROLLMENT", "BIOMETRIC", "DEMOGRAPHIC")
        index_name: Index name (e.g., "AESI", "BRS", "DULI")
        index_value: Optional national average value
    """
    value_html = ""
    if index_value is not None:
        if index_value >= 75:
            color = "#dc2626"
            status = "CRITICAL"
        elif index_value >= 50:
            color = "#f59e0b"
            status = "ELEVATED"
        elif index_value >= 25:
            color = "#eab308"
            status = "MODERATE"
        else:
            color = "#10b981"
            status = "NORMAL"
        
        value_html = f"""
        <div style="text-align: right;">
            <div style="color: #a0aec0; font-size: 10px;">NATIONAL {index_name}</div>
            <div style="color: {color}; font-size: 28px; font-weight: 700;">{index_value:.1f}</div>
            <div style="color: {color}; font-size: 10px; letter-spacing: 1px;">{status}</div>
        </div>
        """
    
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, rgba(26, 41, 65, 0.8) 0%, rgba(12, 25, 41, 0.6) 100%);
                padding: 16px; border-radius: 4px; margin-bottom: 16px; border-left: 4px solid #d4af37;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="color: #d4af37; font-size: 10px; font-weight: 600; letter-spacing: 1px; margin-bottom: 4px;">
                    {domain.upper()} OPERATIONS
                </div>
                <div style="color: #ffffff; font-size: 16px; font-weight: 600;">
                    🗺️ Geographic Situational Awareness - {index_name} Distribution
                </div>
            </div>
            {value_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# QUICK STATS ROW
# =============================================================================
def render_quick_stats_row(stats: List[Dict[str, Union[str, int, float]]]):
    """
    Render a compact row of quick statistics.
    
    Args:
        stats: List of dicts with 'label', 'value', and optional 'color'
    """
    cols = st.columns(len(stats))
    for idx, stat in enumerate(stats):
        with cols[idx]:
            color = stat.get('color', '#ffffff')
            value = stat.get('value', 0)
            label = stat.get('label', '')
            
            # Format value
            if isinstance(value, (int, float)):
                if value >= 1e7:
                    display = f"{value/1e7:.2f} Cr"
                elif value >= 1e5:
                    display = f"{value/1e5:.2f} L"
                elif value >= 1e3:
                    display = f"{value/1e3:.1f} K"
                else:
                    display = f"{value:,.0f}" if isinstance(value, int) else f"{value:.1f}"
            else:
                display = str(value) if value else "N/A"
            
            st.markdown(f"""
            <div style="text-align: center; padding: 8px;">
                <div style="color: {color}; font-size: 22px; font-weight: 700;">{display}</div>
                <div style="color: #a0aec0; font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px;">{label}</div>
            </div>
            """, unsafe_allow_html=True)
