"""
================================================================================
UIDAI COMMAND CENTER - DEMOGRAPHIC INTELLIGENCE
================================================================================
Government Command-and-Control Dashboard for Demographic Update Operations

Design Principles:
- Map-first: DULI Geographic Map as primary visual
- Situation Intelligence Panels for actionable alerts
- Priority Action Tables for operational tasking
- Collapsible methodology (formulas hidden by default)
- Dual Threat Matrix for anomaly correlation
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Optional, List

# Institutional Color Scheme
RISK_COLORS = {'low': '#10b981', 'moderate': '#eab308', 'high': '#f59e0b', 'critical': '#ef4444'}
UIDAI_COLORS = {'navy_dark': '#0c1929', 'navy_light': '#1a2941', 'gold': '#d4af37', 'white': '#ffffff', 'muted': '#a0aec0'}


def render_demographic_kpis(df_demo: pd.DataFrame, risk_df: pd.DataFrame):
    """Render demographic update KPIs with protected display."""
    col1, col2, col3, col4 = st.columns(4)
    
    total = df_demo['total_demo'].sum() if len(df_demo) > 0 and 'total_demo' in df_demo.columns else 0
    districts = df_demo['district'].nunique() if 'district' in df_demo.columns else 0
    high_duli = len(risk_df[risk_df['DULI_normalized'] >= 75]) if 'DULI_normalized' in risk_df.columns else 0
    states = df_demo['state'].nunique() if 'state' in df_demo.columns else 0
    
    def format_num(n):
        if n is None or n == 0: return None
        if n >= 1e7: return f"{n/1e7:.2f} Cr"
        elif n >= 1e5: return f"{n/1e5:.2f} L"
        elif n >= 1e3: return f"{n/1e3:.1f} K"
        return f"{n:,.0f}"
    
    kpi_data = [
        ("TOTAL DEMO UPDATES", format_num(total), "#f59e0b"),
        ("ACTIVE DISTRICTS", format_num(districts), "#3b82f6"),
        ("🔴 HIGH LOAD", str(high_duli) if high_duli > 0 else None, "#ef4444"),
        ("STATES COVERED", str(states) if states > 0 else None, "#10b981")
    ]
    
    for col, (label, value, color) in zip([col1, col2, col3, col4], kpi_data):
        with col:
            if value:
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


def render_ais_analysis(risk_df: pd.DataFrame):
    """Render AIS (Aadhaar Integrity Score) analysis with collapsible methodology."""
    st.markdown("### AIS - Integrity Analysis")
    
    # Situation intelligence panel
    st.markdown("""
    <div style="background: rgba(245, 158, 11, 0.1); border-left: 4px solid #f59e0b; padding: 12px; border-radius: 4px; margin-bottom: 16px;">
        <div style="color: #f59e0b; font-size: 11px; font-weight: 700;">▌ INTEGRITY MONITORING</div>
        <div style="color: #e2e8f0; font-size: 13px; margin-top: 4px;">High AIS districts require investigation. Unusual update patterns may indicate data anomalies.</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Collapsible methodology
    with st.expander("Methodology: AIS Computation", expanded=False):
        st.markdown("""
        **AIS Components:**
        - Security Ratio (50%): Balance between enrollment vs updates
        - Migration Intensity (30%): Demographic mobility patterns  
        - Activity Variance (20%): Operational consistency
        
        **Interpretation:** High AIS = Potential anomaly requiring investigation
        """)
    
    if 'AESI' not in risk_df.columns:
        st.warning("AIS data not available")
        return
    
    # Use AESI as proxy for AIS (they're related)
    df = risk_df.copy()
    df['AIS'] = df['AESI'].fillna(50)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # AIS distribution histogram
        fig = go.Figure(go.Histogram(x=df['AIS'], nbinsx=30, marker=dict(color='#ef4444')))
        fig.add_vline(x=50, line_dash="dash", line_color="#f59e0b", annotation_text="Moderate")
        fig.add_vline(x=75, line_dash="dash", line_color="#ef4444", annotation_text="High Risk")
        fig.update_layout(
            title=dict(text='AIS Score Distribution', font=dict(color='white', size=14)),
            xaxis=dict(title=dict(text='AIS Score', font=dict(color='white')), tickfont=dict(color='white')),
            yaxis=dict(title=dict(text='Districts', font=dict(color='white')), tickfont=dict(color='white')),
            height=350, plot_bgcolor='rgba(26, 41, 66, 0.5)', paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top AIS districts
        top_ais = df.nlargest(10, 'AIS')[['state', 'district', 'AIS']]
        top_ais.columns = ['State', 'District', 'AIS Score']
        top_ais['AIS Score'] = top_ais['AIS Score'].round(1)
        st.markdown("**Top 10 High-AIS Districts:**")
        st.dataframe(top_ais, use_container_width=True, hide_index=True)


def render_security_ratio_analysis(risk_df: pd.DataFrame):
    """Render Security Ratio analysis with collapsible methodology."""
    st.markdown("### Security Ratio Analysis")
    
    # Situation intelligence panel
    st.markdown("""
    <div style="background: rgba(220, 38, 38, 0.1); border-left: 4px solid #dc2626; padding: 12px; border-radius: 4px; margin-bottom: 16px;">
        <div style="color: #dc2626; font-size: 11px; font-weight: 700;">▌ ANOMALY DETECTION</div>
        <div style="color: #e2e8f0; font-size: 13px; margin-top: 4px;">Security Ratio &gt; 3.0 indicates anomalous activity. Districts flagged require field investigation.</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Collapsible methodology
    with st.expander("Methodology: Security Ratio Computation", expanded=False):
        st.markdown("""
        **Formula:** `Security Ratio = Updates / Enrollments`
        
        **Thresholds:**
        - Ratio < 0.5: Low update activity (normal for new areas)
        - Ratio 0.5-1.5: Balanced (healthy ecosystem)
        - Ratio > 1.5: High update activity (potential concern)
        - Ratio > 3.0: Anomalous (requires investigation)
        """)
    
    if 'total_enrollment' not in risk_df.columns or 'total_activity' not in risk_df.columns:
        st.warning("Insufficient data for security ratio")
        return
    
    df = risk_df.copy()
    df['security_ratio'] = df['total_activity'] / (df['total_enrollment'] + 1)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure(go.Histogram(x=df['security_ratio'], nbinsx=30, marker=dict(color='#f59e0b')))
        fig.add_vline(x=1.0, line_dash="dash", line_color="#10b981", annotation_text="Ideal")
        fig.add_vline(x=1.5, line_dash="dash", line_color="#f59e0b", annotation_text="High")
        fig.add_vline(x=3.0, line_dash="dash", line_color="#ef4444", annotation_text="Anomalous")
        fig.update_layout(
            title=dict(text='Security Ratio Distribution', font=dict(color='white', size=14)),
            xaxis=dict(title=dict(text='Security Ratio', font=dict(color='white')), tickfont=dict(color='white')),
            yaxis=dict(title=dict(text='Districts', font=dict(color='white')), tickfont=dict(color='white')),
            height=350, plot_bgcolor='rgba(26, 41, 66, 0.5)', paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        anomalous = df[df['security_ratio'] > 1.5]
        st.markdown(f"**{len(anomalous)} districts with high security ratio (>1.5)**")
        if len(anomalous) > 0:
            display = anomalous.nlargest(10, 'security_ratio')[['state', 'district', 'security_ratio']]
            display.columns = ['State', 'District', 'Security Ratio']
            display['Security Ratio'] = display['Security Ratio'].round(2)
            st.dataframe(display, use_container_width=True, hide_index=True)


def render_dual_threat_matrix(risk_df: pd.DataFrame):
    """Render Dual Threat Matrix with situation intelligence."""
    st.markdown("### Dual Threat Matrix")
    
    # Situation intelligence panel instead of verbose explanation
    st.markdown("""
    <div style="background: rgba(220, 38, 38, 0.1); border-left: 4px solid #dc2626; padding: 12px; border-radius: 4px; margin-bottom: 16px;">
        <div style="color: #dc2626; font-size: 11px; font-weight: 700;">▌ DUAL THREAT ALERT</div>
        <div style="color: #e2e8f0; font-size: 13px; margin-top: 4px;">Upper-right quadrant districts have BOTH high update anomaly AND high overall stress - priority intervention targets.</div>
    </div>
    """, unsafe_allow_html=True)
    
    if 'AESI' not in risk_df.columns:
        st.warning("Insufficient data for dual threat matrix")
        return
    
    df = risk_df.copy()
    df['security_ratio'] = df['total_activity'] / (df['total_enrollment'] + 1) if 'total_enrollment' in df.columns else 1
    
    # Normalize security ratio to 0-100
    df['security_normalized'] = (df['security_ratio'] / (df['security_ratio'].max() + 0.001) * 100).clip(0, 100)
    
    # Quadrant assignment
    aesi_median = df['AESI'].median()
    sec_median = df['security_normalized'].median()
    
    def get_threat_level(row):
        high_aesi = row['AESI'] > aesi_median
        high_sec = row['security_normalized'] > sec_median
        if high_aesi and high_sec: return "DUAL THREAT"
        elif high_aesi: return "High Stress"
        elif high_sec: return "High Anomaly"
        else: return "Normal"
    
    df['threat_level'] = df.apply(get_threat_level, axis=1)
    
    color_map = {
        "DUAL THREAT": "#ef4444",
        "High Stress": "#f59e0b",
        "High Anomaly": "#8b5cf6",
        "Normal": "#10b981"
    }
    
    fig = px.scatter(
        df, x='security_normalized', y='AESI', color='threat_level',
        hover_name='district', hover_data={'state': True},
        color_discrete_map=color_map
    )
    fig.add_hline(y=aesi_median, line_dash="dash", line_color="rgba(255,255,255,0.5)")
    fig.add_vline(x=sec_median, line_dash="dash", line_color="rgba(255,255,255,0.5)")
    
    fig.update_layout(
        title=dict(text='Dual Threat Matrix', font=dict(color='white', size=16)),
        xaxis=dict(title=dict(text='Security Anomaly Score', font=dict(color='white')), tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(title=dict(text='AESI (Stress)', font=dict(color='white')), tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)'),
        legend=dict(title='Threat Level', font=dict(color='white'), bgcolor='rgba(26,41,66,0.8)'),
        height=500, plot_bgcolor='rgba(26, 41, 66, 0.5)', paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Threat distribution
    threat_counts = df['threat_level'].value_counts()
    cols = st.columns(4)
    for idx, (level, count) in enumerate(threat_counts.items()):
        with cols[idx]:
            st.markdown(f"**{level}**<br>{count} districts", unsafe_allow_html=True)


def render_monthly_update_trend(df_demo: pd.DataFrame):
    """Render monthly demographic update trend."""
    if len(df_demo) == 0 or 'date' not in df_demo.columns:
        st.warning("No date data available")
        return
    
    monthly = df_demo.groupby(df_demo['date'].dt.to_period('M'))['total_demo'].sum().reset_index()
    monthly['date'] = monthly['date'].dt.to_timestamp()
    
    fig = go.Figure(go.Scatter(
        x=monthly['date'], y=monthly['total_demo'], name='Demo Updates',
        line=dict(color='#8b5cf6', width=3), fill='tozeroy', fillcolor='rgba(139, 92, 246, 0.2)'
    ))
    fig.update_layout(
        title=dict(text='Monthly Demographic Update Trend', font=dict(color='white', size=16)),
        xaxis=dict(title=dict(text='Month', font=dict(color='white')), tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(title=dict(text='Updates', font=dict(color='white')), tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)'),
        height=400, plot_bgcolor='rgba(26, 41, 66, 0.5)', paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)


def render_state_update_ranking(df_demo: pd.DataFrame):
    """Render state update volume ranking."""
    if len(df_demo) == 0: return
    
    state_data = df_demo.groupby('state')['total_demo'].sum().reset_index()
    state_data = state_data.sort_values('total_demo', ascending=True)
    
    fig = go.Figure(go.Bar(
        y=state_data['state'], x=state_data['total_demo'], orientation='h',
        marker=dict(color='#8b5cf6'), text=state_data['total_demo'].apply(lambda x: f"{x/1e5:.1f}L" if x >= 1e5 else f"{x:,.0f}"),
        textposition='outside', textfont=dict(color='white', size=10)
    ))
    fig.update_layout(
        title=dict(text='State Update Volume', font=dict(color='white', size=16)),
        xaxis=dict(title=dict(text='Total Updates', font=dict(color='white')), tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(title='', tickfont=dict(color='white')),
        height=max(400, len(state_data) * 22), margin=dict(l=10, r=80, t=50, b=10),
        plot_bgcolor='rgba(26, 41, 66, 0.5)', paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# DEMOGRAPHIC DISTRIBUTION ANALYSIS (Moved from statistical.py)
# =============================================================================

# Region mapping for demographic analysis
REGION_MAPPING = {
    'Andhra Pradesh': 'South', 'Telangana': 'South', 'Karnataka': 'South', 'Tamil Nadu': 'South', 'Kerala': 'South',
    'Maharashtra': 'West', 'Gujarat': 'West', 'Goa': 'West', 'Rajasthan': 'North',
    'Uttar Pradesh': 'North', 'Madhya Pradesh': 'Central', 'Bihar': 'East', 'Jharkhand': 'East', 'West Bengal': 'East', 'Odisha': 'East',
    'Punjab': 'North', 'Haryana': 'North', 'Himachal Pradesh': 'North', 'Uttarakhand': 'North', 'NCT of Delhi': 'North', 'Jammu And Kashmir': 'North', 'Ladakh': 'North',
    'Chhattisgarh': 'Central',
    'Assam': 'Northeast', 'Meghalaya': 'Northeast', 'Manipur': 'Northeast', 'Mizoram': 'Northeast', 'Nagaland': 'Northeast', 'Tripura': 'Northeast', 'Arunachal Pradesh': 'Northeast', 'Sikkim': 'Northeast'
}


def render_demographic_distribution(df_demo: pd.DataFrame):
    """Render demographic update distribution histogram."""
    st.markdown("### Demographic Update Distribution")
    
    if len(df_demo) == 0 or 'total_demo' not in df_demo.columns:
        st.info("Demographic distribution data not available")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Demographic histogram
        fig = go.Figure(go.Histogram(
            x=df_demo['total_demo'], nbinsx=30,
            marker=dict(color='#f59e0b', line=dict(color='white', width=1))
        ))
        fig.update_layout(
            title=dict(text='Demographic Update Volume Distribution', font=dict(color='white', size=14)),
            xaxis=dict(title=dict(text='Total Demo Updates', font=dict(color='white')), tickfont=dict(color='white')),
            yaxis=dict(title=dict(text='Frequency', font=dict(color='white')), tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)'),
            height=350, plot_bgcolor='rgba(26, 41, 66, 0.5)', paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top states by demographic updates
        if 'state' in df_demo.columns:
            state_demo = df_demo.groupby('state')['total_demo'].sum().reset_index()
            state_demo = state_demo.nlargest(10, 'total_demo')
            
            fig = go.Figure(go.Bar(
                y=state_demo['state'], x=state_demo['total_demo'], orientation='h',
                marker=dict(color='#f59e0b'),
                text=state_demo['total_demo'].apply(lambda x: f"{x/1e5:.1f}L" if x >= 1e5 else f"{x:,.0f}"),
                textposition='outside', textfont=dict(color='white', size=10)
            ))
            fig.update_layout(
                title=dict(text='Top 10 States by Demo Updates', font=dict(color='white', size=14)),
                xaxis=dict(title=dict(text='Total Updates', font=dict(color='white')), tickfont=dict(color='white')),
                yaxis=dict(title='', tickfont=dict(color='white')),
                height=350, margin=dict(l=10, r=80, t=50, b=10),
                plot_bgcolor='rgba(26, 41, 66, 0.5)', paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)


def render_regional_distribution(df_demo: pd.DataFrame):
    """Render regional demographic distribution pie chart."""
    st.markdown("### Regional Distribution")
    
    if len(df_demo) == 0 or 'state' not in df_demo.columns:
        st.info("Regional data not available")
        return
    
    df = df_demo.copy()
    df['region'] = df['state'].map(REGION_MAPPING).fillna('Other')
    
    region_totals = df.groupby('region')['total_demo'].sum().reset_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart
        fig = go.Figure(go.Pie(
            labels=region_totals['region'], values=region_totals['total_demo'],
            marker=dict(colors=['#1a5276', '#2874a6', '#3b82f6', '#d4af37', '#f59e0b', '#ef4444']),
            textinfo='percent+label', textfont=dict(color='white'), hole=0.4
        ))
        fig.update_layout(
            title=dict(text='Demo Updates by Region', font=dict(color='white', size=14)),
            height=400, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(font=dict(color='white'))
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Bar chart sorted by volume
        region_sorted = region_totals.sort_values('total_demo', ascending=True)
        fig = go.Figure(go.Bar(
            y=region_sorted['region'], x=region_sorted['total_demo'], orientation='h',
            marker=dict(color=['#1a5276', '#2874a6', '#3b82f6', '#d4af37', '#f59e0b', '#ef4444'][:len(region_sorted)]),
            text=region_sorted['total_demo'].apply(lambda x: f"{x/1e5:.1f}L" if x >= 1e5 else f"{x:,.0f}"),
            textposition='outside', textfont=dict(color='white', size=10)
        ))
        fig.update_layout(
            title=dict(text='Regional Volume Comparison', font=dict(color='white', size=14)),
            xaxis=dict(title=dict(text='Total Updates', font=dict(color='white')), tickfont=dict(color='white')),
            yaxis=dict(title='', tickfont=dict(color='white')),
            height=400, margin=dict(l=10, r=80, t=50, b=10),
            plot_bgcolor='rgba(26, 41, 66, 0.5)', paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)


def render_demographic_geographic_view(df_demo: pd.DataFrame, risk_df: pd.DataFrame):
    """Render demographic activity geographic view."""
    st.markdown("### Demographic Update Geographic View")
    
    if 'state' not in df_demo.columns or 'total_demo' not in df_demo.columns:
        st.warning("Insufficient data for geographic visualization")
        return
    
    # State-level aggregation
    state_demo = df_demo.groupby('state').agg({
        'total_demo': 'sum',
        'district': 'nunique' if 'district' in df_demo.columns else 'count'
    }).reset_index()
    state_demo.columns = ['state', 'total_demo', 'district_count']
    
    # Merge with risk data for DULI
    if 'DULI_normalized' in risk_df.columns:
        state_risk = risk_df.groupby('state')['DULI_normalized'].mean().reset_index()
        state_demo = state_demo.merge(state_risk, on='state', how='left')
        state_demo['DULI_normalized'] = state_demo['DULI_normalized'].fillna(50)
    else:
        state_demo['DULI_normalized'] = 50
    
    # Create scatter plot (simulating map)
    fig = px.scatter(
        state_demo, x='district_count', y='total_demo',
        size='total_demo', color='DULI_normalized',
        hover_name='state',
        color_continuous_scale=[[0, '#1a5276'], [0.33, '#2874a6'], [0.66, '#d4af37'], [1, '#c0392b']],
        labels={'district_count': 'Active Districts', 'total_demo': 'Total Demo Updates', 'DULI_normalized': 'DULI Score'}
    )
    fig.update_layout(
        title=dict(text='State Demo Activity (Size = Volume, Color = Load Index)', font=dict(color='white', size=14)),
        xaxis=dict(title=dict(text='Active Districts', font=dict(color='white')), tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(title=dict(text='Total Demo Updates', font=dict(color='white')), tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)'),
        height=450, plot_bgcolor='rgba(26, 41, 66, 0.5)', paper_bgcolor='rgba(0,0,0,0)',
        coloraxis_colorbar=dict(title=dict(text='DULI', font=dict(color='white')), tickfont=dict(color='white'))
    )
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# PREDICTIVE INTELLIGENCE FUNCTIONS - Demographic Domain
# =============================================================================

def render_update_load_forecast(df_demo: pd.DataFrame, risk_df: pd.DataFrame):
    """Render update load forecast map."""
    st.markdown("### Update Load Forecast Map (6-Month Projection)")
    st.info("Predicted demographic update load based on current trends and regional patterns.")
    
    if len(df_demo) == 0 or 'state' not in df_demo.columns:
        st.warning("Insufficient data for forecasting")
        return
    
    # Calculate state-level metrics
    state_demo = df_demo.groupby('state').agg({
        'total_demo': ['sum', 'mean', 'std']
    }).reset_index()
    state_demo.columns = ['state', 'total_demo', 'avg_demo', 'std_demo']
    state_demo['std_demo'] = state_demo['std_demo'].fillna(0)
    
    # Merge with DULI if available
    if 'DULI_normalized' in risk_df.columns:
        state_risk = risk_df.groupby('state')['DULI_normalized'].mean().reset_index()
        state_demo = state_demo.merge(state_risk, on='state', how='left')
        state_demo['current_duli'] = state_demo['DULI_normalized'].fillna(50)
    else:
        state_demo['current_duli'] = 50
    
    # Predict: High volume + high variance = higher future load
    vol_max = state_demo['total_demo'].max() + 1
    state_demo['volume_factor'] = state_demo['total_demo'] / vol_max
    state_demo['predicted_duli'] = (state_demo['current_duli'] * (1 + state_demo['volume_factor'] * 0.18)).clip(0, 100)
    state_demo['duli_change'] = state_demo['predicted_duli'] - state_demo['current_duli']
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Predicted high-load states
        high_load = state_demo.nlargest(12, 'predicted_duli')
        colors = high_load['predicted_duli'].apply(
            lambda x: '#ef4444' if x >= 75 else '#f59e0b' if x >= 50 else '#eab308' if x >= 25 else '#10b981'
        ).tolist()
        
        fig = go.Figure(go.Bar(
            y=high_load['state'], x=high_load['predicted_duli'], orientation='h',
            marker=dict(color=colors),
            text=high_load['predicted_duli'].round(1), textposition='outside',
            textfont=dict(color='white', size=10)
        ))
        fig.add_vline(x=75, line_dash="dash", line_color="#ef4444", opacity=0.7)
        fig.update_layout(
            title=dict(text='Predicted High-Load States', font=dict(color='white', size=14)),
            xaxis=dict(title=dict(text='Predicted DULI', font=dict(color='white')), tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(tickfont=dict(color='white')),
            height=450, margin=dict(l=10, r=60, t=50, b=10),
            plot_bgcolor='rgba(26, 41, 66, 0.5)', paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # States with highest predicted load increase
        top_increase = state_demo.nlargest(10, 'duli_change')
        fig = go.Figure(go.Bar(
            y=top_increase['state'], x=top_increase['duli_change'], orientation='h',
            marker=dict(color='#ef4444'),
            text=top_increase['duli_change'].apply(lambda x: f"+{x:.1f}"),
            textposition='outside', textfont=dict(color='white', size=10)
        ))
        fig.update_layout(
            title=dict(text='Predicted Load Increase by State', font=dict(color='white', size=14)),
            xaxis=dict(title=dict(text='DULI Change', font=dict(color='white')), tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(tickfont=dict(color='white')),
            height=450, margin=dict(l=10, r=60, t=50, b=10),
            plot_bgcolor='rgba(26, 41, 66, 0.5)', paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)


def render_predicted_bottleneck_districts(risk_df: pd.DataFrame):
    """Render predicted bottleneck districts map."""
    st.markdown("### Predicted Bottleneck Districts")
    
    if 'DULI_normalized' not in risk_df.columns:
        if 'AESI' in risk_df.columns:
            risk_df['DULI_normalized'] = risk_df['AESI']
        else:
            st.warning("Load index data required")
            return
    
    df = risk_df.copy()
    df['predicted_duli'] = (df['DULI_normalized'] * 1.15).clip(0, 100)  # 15% increase
    df['will_bottleneck'] = (df['DULI_normalized'] < 75) & (df['predicted_duli'] >= 75)
    df['already_bottleneck'] = df['DULI_normalized'] >= 75
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Already Bottlenecked", f"{df['already_bottleneck'].sum()}")
    with col2:
        st.metric("Will Bottleneck", f"{df['will_bottleneck'].sum()}")
    with col3:
        st.metric("Projected Healthy", f"{(~df['already_bottleneck'] & ~df['will_bottleneck']).sum()}")
    
    # List emerging bottleneck districts
    emerging = df[df['will_bottleneck']][['state', 'district', 'DULI_normalized', 'predicted_duli']].copy()
    if len(emerging) > 0:
        emerging.columns = ['State', 'District', 'Current DULI', 'Predicted DULI']
        emerging = emerging.round(1).sort_values('Predicted DULI', ascending=False).head(15)
        st.markdown("**Districts Projected to Become Bottlenecks:**")
        st.dataframe(emerging, use_container_width=True, hide_index=True)


def render_update_volume_projection(df_demo: pd.DataFrame):
    """Render update volume projection chart."""
    st.markdown("### Update Volume Projection (12-Month Horizon)")
    
    if len(df_demo) == 0 or 'date' not in df_demo.columns:
        st.info("Date information required for volume projection")
        return
    
    # Historical monthly updates
    monthly = df_demo.groupby(df_demo['date'].dt.to_period('M'))['total_demo'].sum().reset_index()
    monthly['date'] = monthly['date'].dt.to_timestamp()
    monthly = monthly.tail(12)
    
    if len(monthly) > 0:
        last_value = monthly['total_demo'].iloc[-1]
        avg_growth = monthly['total_demo'].pct_change().mean()
        avg_growth = 0.025 if pd.isna(avg_growth) else avg_growth  # Default 2.5%
        
        future_dates = pd.date_range(start=monthly['date'].iloc[-1], periods=13, freq='M')[1:]
        predictions = []
        current = last_value
        for i, date in enumerate(future_dates):
            seasonal = 1 + 0.08 * np.sin((i + 1) * np.pi / 6)
            current = current * (1 + avg_growth) * seasonal
            predictions.append({'date': date, 'total_demo': current, 'type': 'Predicted'})
        
        monthly['type'] = 'Historical'
        future_df = pd.DataFrame(predictions)
        combined = pd.concat([monthly, future_df], ignore_index=True)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=combined[combined['type']=='Historical']['date'],
            y=combined[combined['type']=='Historical']['total_demo'],
            name='Historical', mode='lines+markers',
            line=dict(color='#f59e0b', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=combined[combined['type']=='Predicted']['date'],
            y=combined[combined['type']=='Predicted']['total_demo'],
            name='Predicted', mode='lines+markers',
            line=dict(color='#ef4444', width=3, dash='dash'),
            fill='tozeroy', fillcolor='rgba(239, 68, 68, 0.1)'
        ))
        fig.update_layout(
            title=dict(text='Historical vs Predicted Update Volume', font=dict(color='white', size=14)),
            xaxis=dict(title='', tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(title=dict(text='Updates', font=dict(color='white')), tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)'),
            height=400, plot_bgcolor='rgba(26, 41, 66, 0.5)', paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(font=dict(color='white'))
        )
        st.plotly_chart(fig, use_container_width=True)


def render_demographic_dashboard(df_demo: pd.DataFrame, risk_df: pd.DataFrame):
    """Main function to render demographic command center dashboard."""
    
    # Command Center Header
    high_duli = len(risk_df[risk_df['DULI_normalized'] >= 75]) if 'DULI_normalized' in risk_df.columns else 0
    status = "CRITICAL" if high_duli > 10 else "ALERT" if high_duli > 5 else "OPERATIONAL"
    
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, rgba(26, 41, 65, 0.95) 0%, rgba(12, 25, 41, 0.95) 100%);
                padding: 16px 20px; border-radius: 4px; margin-bottom: 16px; border-bottom: 2px solid #f59e0b;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="color: #f59e0b; font-size: 10px; font-weight: 600; letter-spacing: 1px;">DEMOGRAPHIC OPERATIONS</div>
                <div style="color: #ffffff; font-size: 18px; font-weight: 700;">Demographic Intelligence Command</div>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="width: 10px; height: 10px; background: {'#dc2626' if status == 'CRITICAL' else '#f59e0b' if status == 'ALERT' else '#10b981'}; border-radius: 50%;"></span>
                <span style="color: {'#dc2626' if status == 'CRITICAL' else '#f59e0b' if status == 'ALERT' else '#10b981'}; font-size: 11px; font-weight: 600;">{status}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # KPIs
    render_demographic_kpis(df_demo, risk_df)
    
    # Situation Intelligence Panel
    avg_duli = risk_df['DULI_normalized'].mean() if 'DULI_normalized' in risk_df.columns else None
    anomaly_count = len(risk_df[(risk_df.get('AESI', pd.Series([0])) > 50) & (risk_df.get('DULI_normalized', pd.Series([0])) > 50)]) if 'AESI' in risk_df.columns else 0
    
    st.markdown(f"""
    <div style="background: rgba({'220, 38, 38' if high_duli > 5 else '245, 158, 11'}, 0.1); 
                border-left: 4px solid {'#dc2626' if high_duli > 5 else '#f59e0b'}; 
                padding: 16px; border-radius: 4px; margin: 16px 0;">
        <div style="color: {'#dc2626' if high_duli > 5 else '#f59e0b'}; font-size: 11px; font-weight: 700; letter-spacing: 1px; margin-bottom: 12px;">
            ▌ DEMOGRAPHIC SITUATION
        </div>
        <div style="display: flex; gap: 24px; flex-wrap: wrap;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="color: #dc2626; font-size: 28px; font-weight: 700;">{high_duli}</span>
                <span style="color: #e2e8f0; font-size: 13px;">districts with critical DULI (&gt;75) - demographic update bottleneck</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="color: #f59e0b; font-size: 28px; font-weight: 700;">{f'{avg_duli:.1f}' if avg_duli else 'N/A'}</span>
                <span style="color: #e2e8f0; font-size: 13px;">national average DULI score</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # MAP-FIRST tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Map", "Trends", "Distribution", "Priority Action", "Predictive"])
    
    with tab1:
        # MAP-FIRST: DULI Geographic Map
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, rgba(26, 41, 65, 0.8) 0%, rgba(12, 25, 41, 0.6) 100%);
                    padding: 12px 16px; border-radius: 4px; margin-bottom: 12px; border-left: 4px solid #f59e0b;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="color: #f59e0b; font-size: 10px; font-weight: 600; letter-spacing: 1px;">DEMOGRAPHIC OPERATIONS</div>
                    <div style="color: #ffffff; font-size: 15px; font-weight: 600;">DULI Geographic Distribution - Update Load Hotspots</div>
                </div>
                <div style="text-align: right;">
                    <div style="color: #a0aec0; font-size: 10px;">NATIONAL DULI</div>
                    <div style="color: {'#dc2626' if avg_duli and avg_duli >= 75 else '#f59e0b' if avg_duli and avg_duli >= 50 else '#10b981'}; font-size: 24px; font-weight: 700;">{f'{avg_duli:.1f}' if avg_duli else 'N/A'}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        render_demographic_geographic_view(df_demo, risk_df)
        st.markdown("---")
        render_regional_distribution(df_demo)
    
    with tab2:
        # Trends with situation context
        st.markdown("""
        <div style="background: rgba(245, 158, 11, 0.1); border-left: 4px solid #f59e0b; padding: 12px; border-radius: 4px; margin-bottom: 16px;">
            <div style="color: #f59e0b; font-size: 11px; font-weight: 700;">▌ DEMOGRAPHIC TRENDS</div>
            <div style="color: #e2e8f0; font-size: 13px; margin-top: 4px;">Historical update patterns and integrity analysis for capacity planning.</div>
        </div>
        """, unsafe_allow_html=True)
        
        render_monthly_update_trend(df_demo)
        st.markdown("---")
        render_ais_analysis(risk_df)
    
    with tab3:
        # Distribution with situation context
        st.markdown("""
        <div style="background: rgba(245, 158, 11, 0.1); border-left: 4px solid #f59e0b; padding: 12px; border-radius: 4px; margin-bottom: 16px;">
            <div style="color: #f59e0b; font-size: 11px; font-weight: 700;">▌ DISTRIBUTION ANALYSIS</div>
            <div style="color: #e2e8f0; font-size: 13px; margin-top: 4px;">Statistical distribution and security ratio analysis for anomaly detection.</div>
        </div>
        """, unsafe_allow_html=True)
        
        render_demographic_distribution(df_demo)
        st.markdown("---")
        render_security_ratio_analysis(risk_df)
    
    with tab4:
        # PRIORITY ACTION TABLE
        st.markdown("""
        <div style="background: linear-gradient(90deg, rgba(220, 38, 38, 0.15) 0%, rgba(26, 41, 65, 0.8) 100%);
                    padding: 12px 16px; border-radius: 4px; margin-bottom: 16px; border-left: 4px solid #dc2626;">
            <div style="color: #dc2626; font-size: 11px; font-weight: 700; letter-spacing: 1px;">▌ PRIORITY ACTION TABLE</div>
            <div style="color: #e2e8f0; font-size: 13px; margin-top: 4px;">Districts requiring immediate demographic update capacity intervention.</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Priority Action Table for Demographic
        risk_col = 'DULI_normalized' if 'DULI_normalized' in risk_df.columns else 'AESI'
        if risk_col in risk_df.columns:
            priority_df = risk_df.nlargest(10, risk_col)[['state', 'district', risk_col]].copy()
            priority_df['Rank'] = range(1, len(priority_df) + 1)
            
            def get_action(score):
                if score >= 80: return "Emergency processing center"
                elif score >= 60: return "Staff augmentation"
                elif score >= 40: return "Queue management"
                else: return "Monitor"
            
            priority_df['Action'] = priority_df[risk_col].apply(get_action)
            priority_df = priority_df[['Rank', 'state', 'district', risk_col, 'Action']]
            priority_df.columns = ['Rank', 'State', 'District', 'DULI Score', 'Recommended Action']
            priority_df['DULI Score'] = priority_df['DULI Score'].round(1)
            
            st.dataframe(priority_df, use_container_width=True, hide_index=True,
                column_config={'DULI Score': st.column_config.ProgressColumn('DULI Score', format='%.1f', min_value=0, max_value=100)})
        
        st.markdown("---")
        st.markdown("### State Update Rankings")
        render_state_update_ranking(df_demo)
        st.markdown("---")
        st.markdown("### Dual Threat Analysis")
        render_dual_threat_matrix(risk_df)
    
    with tab5:
        # PREDICTIVE INTELLIGENCE
        st.markdown("""
        <div style="background: linear-gradient(90deg, rgba(139, 92, 246, 0.2) 0%, rgba(59, 130, 246, 0.2) 100%); 
                    padding: 15px; border-radius: 4px; border-left: 4px solid #8b5cf6; margin-bottom: 16px;">
            <div style="color: #8b5cf6; font-size: 11px; font-weight: 700; letter-spacing: 1px;">▌ PREDICTIVE INTELLIGENCE</div>
            <div style="color: #e2e8f0; font-size: 13px; margin-top: 4px;">Forward-looking demographic load projections. Use for infrastructure planning, NOT current operations.</div>
        </div>
        """, unsafe_allow_html=True)
        
        render_update_load_forecast(df_demo, risk_df)
        st.markdown("---")
        render_update_volume_projection(df_demo)
        st.markdown("---")
        render_predicted_bottleneck_districts(risk_df)
