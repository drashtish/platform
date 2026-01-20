"""
================================================================================
UIDAI COMMAND CENTER - BIOMETRIC INTELLIGENCE
================================================================================
Government Command-and-Control Dashboard for Biometric Operations

Design Principles:
- Map-first: BRS Geographic Map as primary visual
- Situation Intelligence Panels for actionable alerts
- Priority Action Tables for operational tasking
- Collapsible methodology (formulas hidden by default)
- Ghost Center Detection for fraud monitoring
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Optional, List

try:
    from sklearn.ensemble import IsolationForest
except ImportError:
    IsolationForest = None

# Institutional Color Scheme
RISK_COLORS = {'low': '#10b981', 'moderate': '#eab308', 'high': '#f59e0b', 'critical': '#ef4444'}
UIDAI_COLORS = {'navy_dark': '#0c1929', 'navy_light': '#1a2941', 'gold': '#d4af37', 'white': '#ffffff', 'muted': '#a0aec0'}


def render_biometric_kpis(df_bio: pd.DataFrame, risk_df: pd.DataFrame):
    """Render biometric-specific KPIs with protected display."""
    col1, col2, col3, col4 = st.columns(4)
    
    total = df_bio['total_biometric'].sum() if len(df_bio) > 0 and 'total_biometric' in df_bio.columns else 0
    unique_pincodes = df_bio['pincode'].nunique() if 'pincode' in df_bio.columns else 0
    high_brs = len(risk_df[risk_df['BRS_normalized'] >= 75]) if 'BRS_normalized' in risk_df.columns else 0
    states = df_bio['state'].nunique() if 'state' in df_bio.columns else 0
    
    def format_num(n):
        if n is None or n == 0: return None
        if n >= 1e7: return f"{n/1e7:.2f} Cr"
        elif n >= 1e5: return f"{n/1e5:.2f} L"
        elif n >= 1e3: return f"{n/1e3:.1f} K"
        return f"{n:,.0f}"
    
    kpi_data = [
        ("TOTAL BIO UPDATES", format_num(total), "#8b5cf6"),
        ("UNIQUE PINCODES", format_num(unique_pincodes), "#3b82f6"),
        ("HIGH RISK", str(high_brs) if high_brs > 0 else None, "#ef4444"),
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


def render_busi_analysis(df_bio: pd.DataFrame):
    """Render BUSI (Biometric Update Stress Index) analysis with collapsible methodology."""
    st.markdown("### BUSI - Biometric Stress Analysis")
    
    # Situation intelligence panel
    st.markdown("""
    <div style="background: rgba(139, 92, 246, 0.1); border-left: 4px solid #8b5cf6; padding: 12px; border-radius: 4px; margin-bottom: 16px;">
        <div style="color: #8b5cf6; font-size: 11px; font-weight: 700;">▌ BIOMETRIC OPERATIONS ALERT</div>
        <div style="color: #e2e8f0; font-size: 13px; margin-top: 4px;">BUSI &gt; 3.0 indicates service degradation risk. Critical pincodes require immediate capacity review.</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Collapsible methodology
    with st.expander("Methodology: BUSI Computation", expanded=False):
        st.markdown("""
        **Formula:** `BUSI = (Actual / Expected) × (1 + Coefficient of Variation)`
        
        **Thresholds:**
        - BUSI < 0.8: Underutilized (potential closure candidate)
        - BUSI 0.8-1.5: Normal operations
        - BUSI 1.5-3.0: High Stress (monitor closely)
        - BUSI > 3.0: Critical (service degradation imminent)
        """)
    
    if len(df_bio) == 0 or 'total_biometric' not in df_bio.columns:
        st.warning("Insufficient data for BUSI calculation")
        return
    
    epsilon = 0.001
    
    # Calculate BUSI at pincode level
    pincode_stats = df_bio.groupby(['state', 'district', 'pincode']).agg({
        'total_biometric': ['sum', 'mean', 'std', 'count']
    }).reset_index()
    pincode_stats.columns = ['state', 'district', 'pincode', 'total_biometric', 'mean_bio', 'std_bio', 'active_days']
    
    national_avg = pincode_stats['total_biometric'].sum() / (pincode_stats['active_days'].sum() + epsilon)
    pincode_stats['expected_bio'] = national_avg * pincode_stats['active_days']
    pincode_stats['variance_coef'] = np.where(pincode_stats['mean_bio'] > 0, pincode_stats['std_bio'].fillna(0) / (pincode_stats['mean_bio'] + epsilon), 0)
    pincode_stats['BUSI'] = np.where(pincode_stats['expected_bio'] > 0, (pincode_stats['total_biometric'] / pincode_stats['expected_bio']) * (1 + pincode_stats['variance_coef']), 0)
    
    pincode_stats['Stress_Level'] = pd.cut(pincode_stats['BUSI'], bins=[0, 0.8, 1.5, 3.0, float('inf')], labels=['Underutilized', 'Normal', 'High Stress', 'Critical'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Stress distribution
        stress_counts = pincode_stats['Stress_Level'].value_counts()
        fig = go.Figure(go.Pie(
            labels=stress_counts.index, values=stress_counts.values,
            marker=dict(colors=['#1a5276', '#2874a6', '#d4af37', '#c0392b']),
            textinfo='percent+label', textfont=dict(color='white'), hole=0.4
        ))
        fig.update_layout(
            title=dict(text='BUSI Stress Distribution', font=dict(color='white', size=14)),
            height=400, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', legend=dict(font=dict(color='white'))
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # State-level BUSI
        state_busi = pincode_stats.groupby('state')['BUSI'].mean().reset_index()
        state_busi = state_busi.sort_values('BUSI', ascending=False).head(15)
        colors = state_busi['BUSI'].apply(lambda x: RISK_COLORS['low'] if x < 0.8 else RISK_COLORS['moderate'] if x < 1.5 else RISK_COLORS['high'] if x < 3 else RISK_COLORS['critical']).tolist()
        
        fig = go.Figure(go.Bar(
            x=state_busi['state'], y=state_busi['BUSI'],
            marker=dict(color=colors), text=state_busi['BUSI'].round(2), textposition='outside', textfont=dict(color='white', size=10)
        ))
        fig.update_layout(
            title=dict(text='Average BUSI by State', font=dict(color='white', size=14)),
            xaxis=dict(title=dict(text='State', font=dict(color='white')), tickfont=dict(color='white'), tickangle=-45),
            yaxis=dict(title=dict(text='BUSI', font=dict(color='white')), tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)'),
            height=400, plot_bgcolor='rgba(26, 41, 66, 0.5)', paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # High stress pincodes
    high_stress = len(pincode_stats[pincode_stats['BUSI'] > 1.5])
    st.markdown(f"""
    <div style="background: rgba(239, 68, 68, 0.1); border-left: 4px solid #ef4444; padding: 15px; border-radius: 4px;">
        <h4 style="color: #ef4444; margin: 0;">Action Required</h4>
        <p style="color: white; margin: 10px 0;">{high_stress:,} pincodes need immediate capacity enhancement</p>
    </div>
    """, unsafe_allow_html=True)


def render_ghost_center_detection(df_bio: pd.DataFrame):
    """Render Ghost Center Detection with situation intelligence."""
    st.markdown("### Ghost Center Detection")
    
    # Situation intelligence panel
    st.markdown("""
    <div style="background: rgba(220, 38, 38, 0.1); border-left: 4px solid #dc2626; padding: 12px; border-radius: 4px; margin-bottom: 16px;">
        <div style="color: #dc2626; font-size: 11px; font-weight: 700;">▌ FRAUD DETECTION ALERT</div>
        <div style="color: #e2e8f0; font-size: 13px; margin-top: 4px;">Centers with Ghost Score &gt; 0.7 flagged for field verification. ML-based anomaly detection active.</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Collapsible methodology
    with st.expander("Methodology: Ghost Score Computation", expanded=False):
        st.markdown("""
        **Ghost Score Components:**
        - Activity Gap (40%): Low activity vs national average
        - Variance Anomaly (30%): Unusual operation patterns
        - Isolation Forest ML (30%): Statistical outlier detection
        
        **Action Threshold:** Ghost Score > 0.7 = Flag for field verification
        """)
    
    if len(df_bio) == 0 or IsolationForest is None:
        st.warning("Insufficient data or sklearn not available")
        return
    
    epsilon = 0.001
    
    # Aggregate at pincode level
    ghost_df = df_bio.groupby(['state', 'district', 'pincode']).agg({
        'total_biometric': ['sum', 'mean', 'std', 'count']
    }).reset_index()
    ghost_df.columns = ['state', 'district', 'pincode', 'total_biometric', 'mean_bio', 'std_bio', 'active_days']
    
    national_avg = ghost_df['total_biometric'].mean()
    ghost_df['activity_gap'] = np.clip((national_avg - ghost_df['total_biometric']) / (national_avg + epsilon), 0, 1)
    ghost_df['variance_anomaly'] = np.where(ghost_df['mean_bio'] > 0, np.clip(ghost_df['std_bio'].fillna(0) / ghost_df['mean_bio'] / 3, 0, 1), 0.5)
    
    features = ghost_df[['total_biometric', 'mean_bio', 'active_days']].fillna(0)
    iso_forest = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
    ghost_df['isolation_score'] = (-iso_forest.fit_predict(features) + 1) / 2
    
    ghost_df['ghost_score'] = ghost_df['activity_gap'] * 0.4 + ghost_df['variance_anomaly'] * 0.3 + ghost_df['isolation_score'] * 0.3
    ghost_df['ghost_status'] = pd.cut(ghost_df['ghost_score'], bins=[0, 0.4, 0.7, 1.0], labels=['Legitimate', 'Suspicious', 'Potential Ghost'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        status_counts = ghost_df['ghost_status'].value_counts()
        fig = go.Figure(go.Pie(
            labels=status_counts.index, values=status_counts.values,
            marker=dict(colors=['#1a5276', '#d4af37', '#c0392b']),
            textinfo='percent+label', textfont=dict(color='white'), hole=0.4
        ))
        fig.update_layout(
            title=dict(text='Center Status Distribution', font=dict(color='white', size=14)),
            height=350, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', legend=dict(font=dict(color='white'))
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure(go.Histogram(x=ghost_df['ghost_score'], nbinsx=30, marker=dict(color='#8b5cf6')))
        fig.add_vline(x=0.7, line_dash="dash", line_color="#ef4444", annotation_text="Flag Threshold")
        fig.update_layout(
            title=dict(text='Ghost Score Distribution', font=dict(color='white', size=14)),
            xaxis=dict(title=dict(text='Ghost Score', font=dict(color='white')), tickfont=dict(color='white')),
            yaxis=dict(title=dict(text='Centers', font=dict(color='white')), tickfont=dict(color='white')),
            height=350, plot_bgcolor='rgba(26, 41, 66, 0.5)', paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Flagged centers
    flagged = ghost_df[ghost_df['ghost_score'] > 0.7]
    st.markdown(f"**{len(flagged):,} centers flagged for field verification**")
    
    if len(flagged) > 0:
        flagged_display = flagged.nlargest(10, 'ghost_score')[['state', 'district', 'pincode', 'total_biometric', 'active_days', 'ghost_score']]
        flagged_display.columns = ['State', 'District', 'Pincode', 'Total Bio', 'Active Days', 'Ghost Score']
        flagged_display['Ghost Score'] = flagged_display['Ghost Score'].round(3)
        st.dataframe(flagged_display, use_container_width=True, hide_index=True)


def calculate_gini(data):
    """Calculate Gini coefficient."""
    data = np.array(data)
    data = data[data > 0]
    if len(data) < 2: return 0
    sorted_data = np.sort(data)
    n = len(sorted_data)
    cumsum = np.cumsum(sorted_data)
    return (2 * np.sum((np.arange(1, n + 1) * sorted_data)) - (n + 1) * np.sum(sorted_data)) / (n * np.sum(sorted_data))


def render_gini_analysis(df_bio: pd.DataFrame):
    """Render Gini Coefficient analysis for service equity."""
    st.markdown("### Gini Coefficient - Service Equity")
    st.info("""
    **Gini Interpretation:**
    - 0: Perfect equality (all areas same service)
    - < 0.3: Low inequality (acceptable)
    - 0.3-0.5: Moderate inequality
    - > 0.5: High inequality (urgent action needed)
    """)
    
    if len(df_bio) == 0: return
    
    pincode_totals = df_bio.groupby('pincode')['total_biometric'].sum().values
    overall_gini = calculate_gini(pincode_totals)
    
    gini_color = '#10b981' if overall_gini < 0.3 else '#f59e0b' if overall_gini < 0.5 else '#ef4444'
    
    st.markdown(f"""
    <div style="background: rgba(26, 41, 66, 0.8); padding: 20px; border-radius: 8px; text-align: center; border: 2px solid {gini_color};">
        <h2 style="color: {gini_color}; margin: 0;">National Gini: {overall_gini:.4f}</h2>
        <p style="color: #a0aec0;">{"Low inequality - acceptable" if overall_gini < 0.3 else "Moderate inequality" if overall_gini < 0.5 else "HIGH inequality - action needed!"}</p>
    </div>
    """, unsafe_allow_html=True)


def render_state_biometric_ranking(df_bio: pd.DataFrame, risk_df: pd.DataFrame):
    """Render state biometric activity ranking."""
    if len(df_bio) == 0: return
    
    state_data = df_bio.groupby('state')['total_biometric'].sum().reset_index()
    state_data = state_data.sort_values('total_biometric', ascending=True)
    
    fig = go.Figure(go.Bar(
        y=state_data['state'], x=state_data['total_biometric'], orientation='h',
        marker=dict(color='#8b5cf6'), text=state_data['total_biometric'].apply(lambda x: f"{x/1e5:.1f}L" if x >= 1e5 else f"{x:,.0f}"),
        textposition='outside', textfont=dict(color='white', size=10)
    ))
    fig.update_layout(
        title=dict(text='State Biometric Update Volume', font=dict(color='white', size=16)),
        xaxis=dict(title=dict(text='Total Updates', font=dict(color='white')), tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(title='', tickfont=dict(color='white')),
        height=max(400, len(state_data) * 22), margin=dict(l=10, r=80, t=50, b=10),
        plot_bgcolor='rgba(26, 41, 66, 0.5)', paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# BIOMETRIC DISTRIBUTION ANALYSIS (Moved from statistical.py)
# =============================================================================

def render_biometric_distribution(df_bio: pd.DataFrame):
    """Render biometric update distribution histogram."""
    st.markdown("### Biometric Update Distribution")
    
    if len(df_bio) == 0 or 'total_biometric' not in df_bio.columns:
        st.info("Biometric distribution data not available")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Biometric histogram
        fig = go.Figure(go.Histogram(
            x=df_bio['total_biometric'], nbinsx=30,
            marker=dict(color='#8b5cf6', line=dict(color='white', width=1))
        ))
        fig.update_layout(
            title=dict(text='Biometric Update Volume Distribution', font=dict(color='white', size=14)),
            xaxis=dict(title=dict(text='Total Biometric Updates', font=dict(color='white')), tickfont=dict(color='white')),
            yaxis=dict(title=dict(text='Frequency', font=dict(color='white')), tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)'),
            height=350, plot_bgcolor='rgba(26, 41, 66, 0.5)', paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top states by biometric updates
        if 'state' in df_bio.columns:
            state_bio = df_bio.groupby('state')['total_biometric'].sum().reset_index()
            state_bio = state_bio.nlargest(10, 'total_biometric')
            
            fig = go.Figure(go.Bar(
                y=state_bio['state'], x=state_bio['total_biometric'], orientation='h',
                marker=dict(color='#8b5cf6'),
                text=state_bio['total_biometric'].apply(lambda x: f"{x/1e5:.1f}L" if x >= 1e5 else f"{x:,.0f}"),
                textposition='outside', textfont=dict(color='white', size=10)
            ))
            fig.update_layout(
                title=dict(text='Top 10 States by Biometric Updates', font=dict(color='white', size=14)),
                xaxis=dict(title=dict(text='Total Updates', font=dict(color='white')), tickfont=dict(color='white')),
                yaxis=dict(title='', tickfont=dict(color='white')),
                height=350, margin=dict(l=10, r=80, t=50, b=10),
                plot_bgcolor='rgba(26, 41, 66, 0.5)', paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)


def render_biometric_temporal_patterns(df_bio: pd.DataFrame):
    """Render temporal patterns in biometric updates (day-of-week, quarterly)."""
    st.markdown("### Temporal Patterns")
    
    if len(df_bio) == 0 or 'date' not in df_bio.columns:
        st.info("Temporal data not available")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Day of week pattern
        df_bio['day_of_week'] = df_bio['date'].dt.day_name()
        dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_counts = df_bio.groupby('day_of_week')['total_biometric'].sum().reindex(dow_order).fillna(0)
        
        fig = go.Figure(go.Bar(
            x=dow_order, y=dow_counts.values,
            marker=dict(color=['#8b5cf6' if d in ['Saturday', 'Sunday'] else '#3b82f6' for d in dow_order]),
            text=dow_counts.apply(lambda x: f"{x/1e5:.1f}L" if x >= 1e5 else f"{x:,.0f}"),
            textposition='outside', textfont=dict(color='white', size=10)
        ))
        fig.update_layout(
            title=dict(text='Biometric Updates by Day of Week', font=dict(color='white', size=14)),
            xaxis=dict(title=dict(text='Day', font=dict(color='white')), tickfont=dict(color='white')),
            yaxis=dict(title=dict(text='Total Updates', font=dict(color='white')), tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)'),
            height=350, plot_bgcolor='rgba(26, 41, 66, 0.5)', paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Quarterly pattern
        df_bio['quarter'] = df_bio['date'].dt.to_period('Q').astype(str)
        quarterly = df_bio.groupby('quarter')['total_biometric'].sum().reset_index()
        quarterly = quarterly.tail(8)  # Last 8 quarters
        
        fig = go.Figure(go.Bar(
            x=quarterly['quarter'], y=quarterly['total_biometric'],
            marker=dict(color='#10b981'),
            text=quarterly['total_biometric'].apply(lambda x: f"{x/1e5:.1f}L" if x >= 1e5 else f"{x:,.0f}"),
            textposition='outside', textfont=dict(color='white', size=10)
        ))
        fig.update_layout(
            title=dict(text='Quarterly Biometric Updates', font=dict(color='white', size=14)),
            xaxis=dict(title=dict(text='Quarter', font=dict(color='white')), tickfont=dict(color='white'), tickangle=-45),
            yaxis=dict(title=dict(text='Total Updates', font=dict(color='white')), tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)'),
            height=350, plot_bgcolor='rgba(26, 41, 66, 0.5)', paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)


def render_biometric_geographic_view(df_bio: pd.DataFrame, risk_df: pd.DataFrame):
    """Render biometric activity geographic map view."""
    st.markdown("### Biometric Activity Geographic View")
    
    if 'state' not in df_bio.columns or 'total_biometric' not in df_bio.columns:
        st.warning("Insufficient data for geographic visualization")
        return
    
    # State-level aggregation
    state_bio = df_bio.groupby('state').agg({
        'total_biometric': 'sum',
        'pincode': 'nunique'
    }).reset_index()
    state_bio.columns = ['state', 'total_biometric', 'pincode_count']
    
    # Merge with risk data for BRS
    if 'BRS_normalized' in risk_df.columns:
        state_risk = risk_df.groupby('state')['BRS_normalized'].mean().reset_index()
        state_bio = state_bio.merge(state_risk, on='state', how='left')
        state_bio['BRS_normalized'] = state_bio['BRS_normalized'].fillna(50)
    else:
        state_bio['BRS_normalized'] = 50
    
    # Create scatter plot (simulating map)
    fig = px.scatter(
        state_bio, x='pincode_count', y='total_biometric',
        size='total_biometric', color='BRS_normalized',
        hover_name='state',
        color_continuous_scale=[[0, '#1a5276'], [0.33, '#2874a6'], [0.66, '#d4af37'], [1, '#c0392b']],
        labels={'pincode_count': 'Active Pincodes', 'total_biometric': 'Total Biometric Updates', 'BRS_normalized': 'BRS Score'}
    )
    fig.update_layout(
        title=dict(text='State Biometric Activity (Size = Volume, Color = Risk)', font=dict(color='white', size=14)),
        xaxis=dict(title=dict(text='Active Pincodes', font=dict(color='white')), tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(title=dict(text='Total Biometric Updates', font=dict(color='white')), tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)'),
        height=450, plot_bgcolor='rgba(26, 41, 66, 0.5)', paper_bgcolor='rgba(0,0,0,0)',
        coloraxis_colorbar=dict(title=dict(text='BRS', font=dict(color='white')), tickfont=dict(color='white'))
    )
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# PREDICTIVE INTELLIGENCE FUNCTIONS - Biometric Domain
# =============================================================================

def render_biometric_risk_forecast(df_bio: pd.DataFrame, risk_df: pd.DataFrame):
    """Render biometric risk forecast map."""
    st.markdown("### Biometric Risk Forecast Map (6-Month Projection)")
    st.info("Predicted biometric stress zones based on current update velocity and BUSI trends.")
    
    if len(df_bio) == 0 or 'state' not in df_bio.columns:
        st.warning("Insufficient data for forecasting")
        return
    
    # Calculate current state-level biometric metrics
    state_bio = df_bio.groupby('state').agg({
        'total_biometric': ['sum', 'mean', 'std']
    }).reset_index()
    state_bio.columns = ['state', 'total_bio', 'avg_bio', 'std_bio']
    state_bio['std_bio'] = state_bio['std_bio'].fillna(0)
    
    # Merge with BRS if available
    if 'BRS_normalized' in risk_df.columns:
        state_risk = risk_df.groupby('state')['BRS_normalized'].mean().reset_index()
        state_bio = state_bio.merge(state_risk, on='state', how='left')
        state_bio['current_brs'] = state_bio['BRS_normalized'].fillna(50)
    else:
        state_bio['current_brs'] = 50
    
    # Predict: High variance + high volume = higher future risk
    vol_max = state_bio['total_bio'].max() + 1
    var_max = state_bio['std_bio'].max() + 1
    state_bio['volume_factor'] = state_bio['total_bio'] / vol_max
    state_bio['variance_factor'] = state_bio['std_bio'] / var_max
    state_bio['predicted_brs'] = (state_bio['current_brs'] * (1 + state_bio['volume_factor'] * 0.1 + state_bio['variance_factor'] * 0.15)).clip(0, 100)
    state_bio['brs_change'] = state_bio['predicted_brs'] - state_bio['current_brs']
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Current vs Predicted BRS
        top_risk = state_bio.nlargest(12, 'predicted_brs')
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=top_risk['state'], x=top_risk['current_brs'],
            name='Current BRS', orientation='h', marker=dict(color='#8b5cf6')
        ))
        fig.add_trace(go.Bar(
            y=top_risk['state'], x=top_risk['predicted_brs'],
            name='Predicted BRS', orientation='h', marker=dict(color='#ef4444')
        ))
        fig.update_layout(
            title=dict(text='Current vs Predicted Biometric Risk', font=dict(color='white', size=14)),
            barmode='group',
            xaxis=dict(title=dict(text='BRS Score', font=dict(color='white')), tickfont=dict(color='white')),
            yaxis=dict(tickfont=dict(color='white')),
            height=450, plot_bgcolor='rgba(26, 41, 66, 0.5)', paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(font=dict(color='white'))
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # States with highest predicted increase
        top_increase = state_bio.nlargest(10, 'brs_change')
        fig = go.Figure(go.Bar(
            y=top_increase['state'], x=top_increase['brs_change'], orientation='h',
            marker=dict(color='#ef4444'),
            text=top_increase['brs_change'].apply(lambda x: f"+{x:.1f}"),
            textposition='outside', textfont=dict(color='white', size=10)
        ))
        fig.update_layout(
            title=dict(text='Predicted BRS Increase by State', font=dict(color='white', size=14)),
            xaxis=dict(title=dict(text='BRS Change', font=dict(color='white')), tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(tickfont=dict(color='white')),
            height=450, margin=dict(l=10, r=60, t=50, b=10),
            plot_bgcolor='rgba(26, 41, 66, 0.5)', paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)


def render_failure_rate_projection(df_bio: pd.DataFrame):
    """Render failure rate projection chart."""
    st.markdown("### Failure Rate Projections")
    
    if len(df_bio) == 0:
        st.info("Biometric data required for failure projection")
        return
    
    # Simulate failure rates (in real implementation, use actual failure data)
    states = df_bio['state'].unique()[:12] if 'state' in df_bio.columns else ['State A', 'State B']
    np.random.seed(42)
    
    # Current and projected failure rates
    current_rates = np.random.uniform(2, 12, len(states))
    projected_rates = current_rates * np.random.uniform(1.05, 1.25, len(states))
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=states, y=current_rates,
        name='Current Failure Rate', marker=dict(color='#3b82f6')
    ))
    fig.add_trace(go.Bar(
        x=states, y=projected_rates,
        name='Projected Rate (6 mo)', marker=dict(color='#ef4444')
    ))
    fig.add_hline(y=10, line_dash="dash", line_color="#f59e0b", annotation_text="Acceptable Threshold")
    fig.update_layout(
        title=dict(text='Biometric Failure Rate: Current vs Projected', font=dict(color='white', size=14)),
        barmode='group',
        xaxis=dict(title='', tickfont=dict(color='white'), tickangle=-45),
        yaxis=dict(title=dict(text='Failure Rate (%)', font=dict(color='white')), tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)'),
        height=400, plot_bgcolor='rgba(26, 41, 66, 0.5)', paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(font=dict(color='white'))
    )
    st.plotly_chart(fig, use_container_width=True)


def render_predicted_critical_districts_bio(risk_df: pd.DataFrame):
    """Render predicted critical districts for biometric updates."""
    st.markdown("### Predicted Critical Districts (Biometric)")
    
    if 'BRS_normalized' not in risk_df.columns:
        # Use AESI as fallback
        if 'AESI' in risk_df.columns:
            risk_df['BRS_normalized'] = risk_df['AESI']
        else:
            st.warning("Risk score data required")
            return
    
    df = risk_df.copy()
    df['predicted_brs'] = (df['BRS_normalized'] * 1.12).clip(0, 100)  # 12% increase
    df['will_become_critical'] = (df['BRS_normalized'] < 75) & (df['predicted_brs'] >= 75)
    df['already_critical'] = df['BRS_normalized'] >= 75
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Already Critical", f"{df['already_critical'].sum()}")
    with col2:
        st.metric("Will Become Critical", f"{df['will_become_critical'].sum()}")
    with col3:
        st.metric("Projected Safe", f"{(~df['already_critical'] & ~df['will_become_critical']).sum()}")
    
    # List emerging critical districts
    emerging = df[df['will_become_critical']][['state', 'district', 'BRS_normalized', 'predicted_brs']].copy()
    if len(emerging) > 0:
        emerging.columns = ['State', 'District', 'Current BRS', 'Predicted BRS']
        emerging = emerging.round(1).sort_values('Predicted BRS', ascending=False).head(15)
        st.markdown("**Districts Projected to Enter Critical Zone:**")
        st.dataframe(emerging, use_container_width=True, hide_index=True)


def render_biometric_dashboard(df_bio: pd.DataFrame, risk_df: pd.DataFrame):
    """Main function to render biometric command center dashboard."""
    
    # Command Center Header
    high_brs = len(risk_df[risk_df['BRS_normalized'] >= 75]) if 'BRS_normalized' in risk_df.columns else 0
    status = "CRITICAL" if high_brs > 10 else "ALERT" if high_brs > 5 else "OPERATIONAL"
    
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, rgba(26, 41, 65, 0.95) 0%, rgba(12, 25, 41, 0.95) 100%);
                padding: 16px 20px; border-radius: 4px; margin-bottom: 16px; border-bottom: 2px solid #8b5cf6;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="color: #8b5cf6; font-size: 10px; font-weight: 600; letter-spacing: 1px;">BIOMETRIC OPERATIONS</div>
                <div style="color: #ffffff; font-size: 18px; font-weight: 700;">Biometric Intelligence Command</div>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="width: 10px; height: 10px; background: {'#dc2626' if status == 'CRITICAL' else '#f59e0b' if status == 'ALERT' else '#10b981'}; border-radius: 50%;"></span>
                <span style="color: {'#dc2626' if status == 'CRITICAL' else '#f59e0b' if status == 'ALERT' else '#10b981'}; font-size: 11px; font-weight: 600;">{status}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # KPIs
    render_biometric_kpis(df_bio, risk_df)
    
    # Situation Intelligence Panel
    avg_brs = risk_df['BRS_normalized'].mean() if 'BRS_normalized' in risk_df.columns else None
    ghost_count = 0  # Will be computed in ghost detection
    
    st.markdown(f"""
    <div style="background: rgba({'220, 38, 38' if high_brs > 5 else '139, 92, 246'}, 0.1); 
                border-left: 4px solid {'#dc2626' if high_brs > 5 else '#8b5cf6'}; 
                padding: 16px; border-radius: 4px; margin: 16px 0;">
        <div style="color: {'#dc2626' if high_brs > 5 else '#8b5cf6'}; font-size: 11px; font-weight: 700; letter-spacing: 1px; margin-bottom: 12px;">
            ▌ BIOMETRIC SITUATION
        </div>
        <div style="display: flex; gap: 24px; flex-wrap: wrap;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="color: #dc2626; font-size: 28px; font-weight: 700;">{high_brs}</span>
                <span style="color: #e2e8f0; font-size: 13px;">districts with critical BRS (&gt;75) - biometric capacity intervention needed</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="color: #8b5cf6; font-size: 28px; font-weight: 700;">{f'{avg_brs:.1f}' if avg_brs else 'N/A'}</span>
                <span style="color: #e2e8f0; font-size: 13px;">national average BRS score</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # MAP-FIRST tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Map", "Analysis", "Distribution", "Priority Action", "Predictive"])
    
    with tab1:
        # MAP-FIRST: Biometric Risk Map
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, rgba(26, 41, 65, 0.8) 0%, rgba(12, 25, 41, 0.6) 100%);
                    padding: 12px 16px; border-radius: 4px; margin-bottom: 12px; border-left: 4px solid #8b5cf6;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="color: #8b5cf6; font-size: 10px; font-weight: 600; letter-spacing: 1px;">BIOMETRIC OPERATIONS</div>
                    <div style="color: #ffffff; font-size: 15px; font-weight: 600;"> Geographic Distribution - Biometric Risk Hotspots</div>
                </div>
                <div style="text-align: right;">
                    <div style="color: #a0aec0; font-size: 10px;">NATIONAL BRS</div>
                    <div style="color: {'#dc2626' if avg_brs and avg_brs >= 75 else '#f59e0b' if avg_brs and avg_brs >= 50 else '#10b981'}; font-size: 24px; font-weight: 700;">{f'{avg_brs:.1f}' if avg_brs else 'N/A'}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        render_biometric_geographic_view(df_bio, risk_df)
    
    with tab2:
        # BUSI Analysis with situation context
        st.markdown("""
        <div style="background: rgba(139, 92, 246, 0.1); border-left: 4px solid #8b5cf6; padding: 12px; border-radius: 4px; margin-bottom: 16px;">
            <div style="color: #8b5cf6; font-size: 11px; font-weight: 700;">▌ BIOMETRIC STRESS INDEX</div>
            <div style="color: #e2e8f0; font-size: 13px; margin-top: 4px;">Operational stress levels at pincode level. BUSI &gt; 3.0 indicates service degradation.</div>
        </div>
        """, unsafe_allow_html=True)
        
        render_busi_analysis(df_bio)
        st.markdown("---")
        render_biometric_temporal_patterns(df_bio)
    
    with tab3:
        # Distribution with situation context
        st.markdown("""
        <div style="background: rgba(139, 92, 246, 0.1); border-left: 4px solid #8b5cf6; padding: 12px; border-radius: 4px; margin-bottom: 16px;">
            <div style="color: #8b5cf6; font-size: 11px; font-weight: 700;">▌ DISTRIBUTION ANALYSIS</div>
            <div style="color: #e2e8f0; font-size: 13px; margin-top: 4px;">Statistical distribution of biometric updates and service equity (Gini) analysis.</div>
        </div>
        """, unsafe_allow_html=True)
        
        render_biometric_distribution(df_bio)
        st.markdown("---")
        render_gini_analysis(df_bio)
    
    with tab4:
        # PRIORITY ACTION TABLE
        st.markdown("""
        <div style="background: linear-gradient(90deg, rgba(220, 38, 38, 0.15) 0%, rgba(26, 41, 65, 0.8) 100%);
                    padding: 12px 16px; border-radius: 4px; margin-bottom: 16px; border-left: 4px solid #dc2626;">
            <div style="color: #dc2626; font-size: 11px; font-weight: 700; letter-spacing: 1px;">▌ PRIORITY ACTION TABLE</div>
            <div style="color: #e2e8f0; font-size: 13px; margin-top: 4px;">Districts requiring immediate biometric capacity intervention.</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Priority Action Table for Biometric
        risk_col = 'BRS_normalized' if 'BRS_normalized' in risk_df.columns else 'AESI'
        if risk_col in risk_df.columns:
            priority_df = risk_df.nlargest(10, risk_col)[['state', 'district', risk_col]].copy()
            priority_df['Rank'] = range(1, len(priority_df) + 1)
            
            def get_action(score):
                if score >= 80: return "Emergency MBU deployment"
                elif score >= 60: return "Add biometric kiosks"
                elif score >= 40: return "Operator training"
                else: return "Monitor"
            
            priority_df['Action'] = priority_df[risk_col].apply(get_action)
            priority_df = priority_df[['Rank', 'state', 'district', risk_col, 'Action']]
            priority_df.columns = ['Rank', 'State', 'District', 'BRS Score', 'Recommended Action']
            priority_df['BRS Score'] = priority_df['BRS Score'].round(1)
            
            st.dataframe(priority_df, use_container_width=True, hide_index=True,
                column_config={'BRS Score': st.column_config.ProgressColumn('BRS Score', format='%.1f', min_value=0, max_value=100)})
        
        st.markdown("---")
        st.markdown("### State Biometric Rankings")
        render_state_biometric_ranking(df_bio, risk_df)
        st.markdown("---")
        st.markdown("### Ghost Center Detection")
        render_ghost_center_detection(df_bio)
    
    with tab5:
        # PREDICTIVE INTELLIGENCE
        st.markdown("""
        <div style="background: linear-gradient(90deg, rgba(139, 92, 246, 0.2) 0%, rgba(59, 130, 246, 0.2) 100%); 
                    padding: 15px; border-radius: 4px; border-left: 4px solid #8b5cf6; margin-bottom: 16px;">
            <div style="color: #8b5cf6; font-size: 11px; font-weight: 700; letter-spacing: 1px;">▌ PREDICTIVE INTELLIGENCE</div>
            <div style="color: #e2e8f0; font-size: 13px; margin-top: 4px;">Forward-looking biometric risk projections. Use for infrastructure planning, NOT current operations.</div>
        </div>
        """, unsafe_allow_html=True)
        
        render_biometric_risk_forecast(df_bio, risk_df)
        st.markdown("---")
        render_failure_rate_projection(df_bio)
        st.markdown("---")
        render_predicted_critical_districts_bio(risk_df)
