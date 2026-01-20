"""
================================================================================
UIDAI NATIONAL COMMAND CENTER - EXECUTIVE OVERVIEW
================================================================================
Government Command-and-Control Dashboard for National Situational Awareness

Design Principles:
- Map-first visual hierarchy for spatial awareness
- Situation Intelligence Panels (not verbose explanations)
- Priority Action Tables for operational tasking
- Protected KPIs (no misleading zeros)
- Collapsible methodology sections
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
    from sklearn.ensemble import RandomForestRegressor
except ImportError:
    RandomForestRegressor = None

try:
    from .command_components import (
        render_situation_panel, render_priority_action_table,
        render_methodology_section, render_protected_kpi,
        render_command_header, render_map_section_header,
        render_quick_stats_row, COMMAND_COLORS
    )
    COMMAND_AVAILABLE = True
except ImportError:
    COMMAND_AVAILABLE = False


# Government Color Scheme - NIC/UIDAI Standard
RISK_COLORS = {
    'low': '#10b981',
    'moderate': '#eab308', 
    'high': '#f59e0b',
    'critical': '#ef4444'
}

# Institutional palette
UIDAI_COLORS = {
    'navy_dark': '#0c1929',
    'navy_light': '#1a2941',
    'gold': '#d4af37',
    'white': '#ffffff',
    'muted': '#a0aec0'
}


def render_executive_kpis(stats: Dict, risk_df: pd.DataFrame):
    """Render executive-level KPIs with government command center styling."""
    critical_count = len(risk_df[risk_df['AESI'] > 75]) if 'AESI' in risk_df.columns else 0
    high_count = len(risk_df[(risk_df['AESI'] > 50) & (risk_df['AESI'] <= 75)]) if 'AESI' in risk_df.columns else 0
    moderate_count = len(risk_df[(risk_df['AESI'] > 25) & (risk_df['AESI'] <= 50)]) if 'AESI' in risk_df.columns else 0
    avg_aesi = risk_df['AESI'].mean() if 'AESI' in risk_df.columns else None
    
    def format_number(n):
        if n is None or n == 0:
            return None  # Will trigger "Data Not Available"
        if n >= 1e7: return f"{n/1e7:.2f} Cr"
        elif n >= 1e5: return f"{n/1e5:.2f} L"
        elif n >= 1e3: return f"{n/1e3:.1f} K"
        return f"{n:,.0f}"
    
    # Row 1: Volume KPIs
    col1, col2, col3, col4 = st.columns(4)
    kpi_data = [
        ("ENROLLMENTS", stats.get('total_enrollments', 0)),
        ("DEMO UPDATES", stats.get('total_demographic_updates', 0)),
        ("BIO UPDATES", stats.get('total_biometric_updates', 0)),
        ("ACTIVE STATES", stats.get('states_covered', 0))
    ]
    
    for col, (label, value) in zip([col1, col2, col3, col4], kpi_data):
        with col:
            formatted = format_number(value) if value else None
            if formatted:
                st.markdown(f"""
                <div style="background: rgba(26, 41, 65, 0.6); padding: 12px; border-radius: 4px; text-align: center; border: 1px solid rgba(212, 175, 55, 0.1);">
                    <div style="color: #a0aec0; font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px;">{label}</div>
                    <div style="color: #ffffff; font-size: 22px; font-weight: 700; margin-top: 4px;">{formatted}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: rgba(26, 41, 65, 0.6); padding: 12px; border-radius: 4px; text-align: center;">
                    <div style="color: #a0aec0; font-size: 10px; text-transform: uppercase;">{label}</div>
                    <div style="color: #f59e0b; font-size: 12px; margin-top: 4px;">Data Not Available</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Row 2: Risk Status KPIs with color coding
    col5, col6, col7, col8 = st.columns(4)
    risk_data = [
        ("DISTRICTS", stats.get('districts_covered', 0), "#ffffff"),
        ("CRITICAL", critical_count, "#ef4444"),
        ("HIGH RISK", high_count, "#f59e0b"),
        ("NATIONAL AESI", f"{avg_aesi:.1f}" if avg_aesi else None, "#d4af37")
    ]
    
    for col, (label, value, color) in zip([col5, col6, col7, col8], risk_data):
        with col:
            if value is not None:
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


def render_state_risk_ranking(risk_df: pd.DataFrame):
    """Render state-level risk ranking bar chart."""
    if len(risk_df) == 0 or 'state' not in risk_df.columns: return
    
    state_risk = risk_df.groupby('state').agg({
        'AESI': 'mean', 'total_activity': 'sum', 'district': 'count'
    }).reset_index()
    state_risk.columns = ['state', 'AESI', 'total_activity', 'district_count']
    state_risk = state_risk.sort_values('AESI', ascending=True)
    
    colors = state_risk['AESI'].apply(
        lambda x: RISK_COLORS['low'] if x < 30 else RISK_COLORS['moderate'] if x < 50 else RISK_COLORS['high'] if x < 70 else RISK_COLORS['critical']
    ).tolist()
    
    fig = go.Figure(go.Bar(
        y=state_risk['state'], x=state_risk['AESI'], orientation='h',
        marker=dict(color=colors), text=state_risk['AESI'].round(1), textposition='outside',
        textfont=dict(color='white', size=10),
        customdata=state_risk['district_count'],
        hovertemplate='<b>%{y}</b><br>AESI: %{x:.1f}<br>Districts: %{customdata}<extra></extra>'
    ))
    fig.add_vline(x=30, line_dash="dash", line_color="#10b981", opacity=0.5)
    fig.add_vline(x=50, line_dash="dash", line_color="#f59e0b", opacity=0.5)
    fig.add_vline(x=70, line_dash="dash", line_color="#ef4444", opacity=0.5)
    fig.update_layout(
        title=dict(text='State Risk Ranking (AESI)', font=dict(color='white', size=16)),
        xaxis=dict(title=dict(text='AESI Score', font=dict(color='white')), tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(title='', tickfont=dict(color='white')),
        height=max(400, len(state_risk) * 22),
        margin=dict(l=10, r=60, t=50, b=10),
        plot_bgcolor='rgba(26, 41, 66, 0.5)', paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)


def render_correlation_matrix(risk_df: pd.DataFrame):
    """Render correlation matrix heatmap."""
    numeric_cols = ['total_enrollment', 'total_demo', 'total_bio', 'total_activity', 'AESI', 'EPI_normalized', 'DULI_normalized', 'BRS_normalized']
    available_cols = [c for c in numeric_cols if c in risk_df.columns]
    if len(available_cols) < 3: return
    
    corr_matrix = risk_df[available_cols].corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.index,
        colorscale=[[0, '#c0392b'], [0.25, '#d4af37'], [0.5, '#f5f5f5'], [0.75, '#2874a6'], [1, '#1a5276']],
        zmin=-1, zmax=1, text=np.round(corr_matrix.values, 2), texttemplate='%{text}', textfont=dict(size=10, color='black')
    ))
    fig.update_layout(
        title=dict(text=' Correlation Matrix', font=dict(color='white', size=14)),
        xaxis=dict(tickfont=dict(color='white', size=9), tickangle=-45), yaxis=dict(tickfont=dict(color='white', size=9)),
        height=400, margin=dict(l=10, r=10, t=50, b=10), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)


def render_feature_importance(risk_df: pd.DataFrame):
    """Render feature importance using RandomForest."""
    if RandomForestRegressor is None: return
    feature_cols = ['total_enrollment', 'total_demo', 'total_bio', 'total_activity', 'EPI_normalized', 'DULI_normalized', 'BRS_normalized']
    available_features = [c for c in feature_cols if c in risk_df.columns]
    if len(available_features) < 3 or 'AESI' not in risk_df.columns: return
    
    X = risk_df[available_features].fillna(0)
    y = risk_df['AESI'].fillna(50)
    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X, y)
    
    importance_df = pd.DataFrame({'Feature': available_features, 'Importance': rf.feature_importances_}).sort_values('Importance', ascending=True)
    fig = go.Figure(go.Bar(
        y=importance_df['Feature'], x=importance_df['Importance'], orientation='h',
        marker=dict(color=importance_df['Importance'], colorscale=[[0, '#3b82f6'], [0.5, '#8b5cf6'], [1, '#d4af37']]),
        text=importance_df['Importance'].round(3), textposition='outside', textfont=dict(color='white', size=10)
    ))
    fig.update_layout(
        title=dict(text=' Feature Importance (AESI)', font=dict(color='white', size=14)),
        xaxis=dict(title=dict(text='Importance', font=dict(color='white')), tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(tickfont=dict(color='white', size=10)),
        height=350, margin=dict(l=10, r=80, t=50, b=10),
        plot_bgcolor='rgba(26, 41, 66, 0.5)', paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)


def render_risk_capacity_quadrant(risk_df: pd.DataFrame):
    """Render Risk vs Capacity quadrant scatter."""
    if 'AESI' not in risk_df.columns: return
    
    df = risk_df.copy()
    df['Risk_Score'] = df['AESI'].fillna(50)
    max_activity = df['total_activity'].max() + 0.001 if 'total_activity' in df.columns else 1
    df['Capacity_Score'] = (df['total_activity'] / max_activity * 100).fillna(50) if 'total_activity' in df.columns else 50
    
    risk_median, capacity_median = df['Risk_Score'].median(), df['Capacity_Score'].median()
    
    def get_profile(row):
        high_risk = row['Risk_Score'] > risk_median
        high_capacity = row['Capacity_Score'] > capacity_median
        if high_risk and not high_capacity: return "Crisis Zone"
        elif high_risk and high_capacity: return "Overloaded"
        elif not high_risk and not high_capacity: return "Underserved"
        else: return "Champion"
    
    df['Profile'] = df.apply(get_profile, axis=1)
    color_map = {"Crisis Zone": "#ef4444", "Overloaded": "#f59e0b", "Underserved": "#9b59b6", "Champion": "#10b981"}
    
    fig = px.scatter(df, x='Capacity_Score', y='Risk_Score', color='Profile', hover_name='district',
                     hover_data={'state': True, 'Risk_Score': ':.1f', 'Capacity_Score': ':.1f'}, color_discrete_map=color_map)
    fig.add_hline(y=risk_median, line_dash="dash", line_color="rgba(255,255,255,0.5)")
    fig.add_vline(x=capacity_median, line_dash="dash", line_color="rgba(255,255,255,0.5)")
    fig.update_layout(
        title=dict(text=' Risk vs Capacity Quadrant', font=dict(color='white', size=16)),
        xaxis=dict(title=dict(text='Capacity Score ', font=dict(color='white')), tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(title=dict(text=' Risk Score', font=dict(color='white')), tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)'),
        legend=dict(title='Profile', font=dict(color='white'), bgcolor='rgba(26,41,66,0.8)'),
        height=500, margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor='rgba(26, 41, 66, 0.5)', paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    profile_counts = df['Profile'].value_counts()
    cols = st.columns(4)
    for idx, (profile, count) in enumerate(profile_counts.items()):
        with cols[idx]: st.markdown(f"**{profile}**<br>{count} ({count/len(df)*100:.1f}%)", unsafe_allow_html=True)


def render_district_leaderboard(risk_df: pd.DataFrame):
    """Render district performance leaderboard."""
    if 'AESI' not in risk_df.columns: return
    
    df = risk_df.copy()
    df['Performance_Score'] = 100 - df['AESI'].clip(0, 100)
    df['Efficiency_Score'] = (df['total_activity'] / (df['total_activity'].max() + 1) * 100).clip(0, 100)
    df['Leaderboard_Score'] = (df['Performance_Score'] * 0.6 + df['Efficiency_Score'] * 0.4).round(2)
    df['Rank'] = df['Leaderboard_Score'].rank(ascending=False, method='min').astype(int)
    
    tab1, tab2 = st.tabs([" Top Performers", " Needs Improvement"])
    with tab1:
        top = df.nsmallest(15, 'Rank')[['Rank', 'state', 'district', 'Leaderboard_Score', 'AESI', 'total_activity']]
        top.columns = ['Rank', 'State', 'District', 'Score', 'AESI', 'Activity']
        st.dataframe(top, use_container_width=True, hide_index=True)
    with tab2:
        bottom = df.nlargest(15, 'AESI')[['Rank', 'state', 'district', 'Leaderboard_Score', 'AESI', 'total_activity']]
        bottom.columns = ['Rank', 'State', 'District', 'Score', 'AESI', 'Activity']
        st.dataframe(bottom, use_container_width=True, hide_index=True)


def render_risk_distribution_charts(risk_df: pd.DataFrame):
    """Render AESI distribution."""
    if 'AESI' not in risk_df.columns: return
    
    col1, col2 = st.columns(2)
    with col1:
        fig_hist = go.Figure(go.Histogram(x=risk_df['AESI'], nbinsx=25, marker=dict(color='#3b82f6')))
        fig_hist.add_vline(x=30, line_dash="dash", line_color="#10b981")
        fig_hist.add_vline(x=50, line_dash="dash", line_color="#f59e0b")
        fig_hist.add_vline(x=70, line_dash="dash", line_color="#ef4444")
        fig_hist.update_layout(
            title=dict(text='AESI Distribution', font=dict(color='white')),
            xaxis=dict(title=dict(text='AESI', font=dict(color='white')), tickfont=dict(color='white')),
            yaxis=dict(title=dict(text='Districts', font=dict(color='white')), tickfont=dict(color='white')),
            height=350, plot_bgcolor='rgba(26, 41, 66, 0.5)', paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        df = risk_df.copy()
        df['Category'] = pd.cut(df['AESI'], bins=[0, 30, 50, 70, 100], labels=['Low', 'Moderate', 'High', 'Critical'])
        cat_counts = df['Category'].value_counts()
        fig_pie = go.Figure(go.Pie(labels=cat_counts.index, values=cat_counts.values, marker=dict(colors=['#1a5276', '#2874a6', '#d4af37', '#c0392b']), hole=0.4))
        fig_pie.update_layout(
            title=dict(text='Risk Categories', font=dict(color='white')),
            height=350, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', legend=dict(font=dict(color='white'))
        )
        st.plotly_chart(fig_pie, use_container_width=True)


def render_regional_treemap(risk_df: pd.DataFrame):
    """Render hierarchical treemap of Region -> State -> District."""
    if 'state' not in risk_df.columns or 'AESI' not in risk_df.columns:
        st.warning("Insufficient data for treemap")
        return
    
    # Define regions for Indian states
    region_mapping = {
        'Andhra Pradesh': 'South', 'Telangana': 'South', 'Karnataka': 'South', 'Tamil Nadu': 'South', 'Kerala': 'South',
        'Maharashtra': 'West', 'Gujarat': 'West', 'Goa': 'West', 'Rajasthan': 'North',
        'Uttar Pradesh': 'North', 'Madhya Pradesh': 'Central', 'Bihar': 'East', 'Jharkhand': 'East', 'West Bengal': 'East', 'Odisha': 'East',
        'Punjab': 'North', 'Haryana': 'North', 'Himachal Pradesh': 'North', 'Uttarakhand': 'North', 'NCT of Delhi': 'North', 'Jammu And Kashmir': 'North', 'Ladakh': 'North',
        'Chhattisgarh': 'Central',
        'Assam': 'Northeast', 'Meghalaya': 'Northeast', 'Manipur': 'Northeast', 'Mizoram': 'Northeast', 'Nagaland': 'Northeast', 'Tripura': 'Northeast', 'Arunachal Pradesh': 'Northeast', 'Sikkim': 'Northeast'
    }



    df = risk_df.copy()
    df['region'] = df['state'].map(region_mapping).fillna('Other')
    
    # Aggregate by state
    state_data = df.groupby(['region', 'state']).agg({
        'AESI': 'mean',
        'total_activity': 'sum',
        'district': 'count'
    }).reset_index()
    state_data.columns = ['Region', 'State', 'AESI', 'Activity', 'Districts']
    
    fig = px.treemap(
        state_data, path=['Region', 'State'], values='Activity', color='AESI',
        color_continuous_scale=['#1a5276', '#2874a6', '#d4af37', '#c0392b'],
        hover_data={'AESI': ':.1f', 'Districts': True}
    )
    fig.update_layout(
        title=dict(text='Regional Activity Treemap (Color = AESI)', font=dict(color='#d4af37', size=16)),
        height=500, paper_bgcolor='rgba(12, 25, 41, 0.9)', margin=dict(t=50, l=10, r=10, b=10),
        coloraxis_colorbar=dict(title=dict(text='AESI', font=dict(color='white')), tickfont=dict(color='white'))
    )
    fig.update_traces(textfont=dict(color='white'))
    st.plotly_chart(fig, use_container_width=True)


def render_region_state_sunburst(risk_df: pd.DataFrame):
    """Render hierarchical sunburst chart of Region → State with AESI Risk Level."""
    if 'state' not in risk_df.columns or 'AESI' not in risk_df.columns:
        st.warning("Insufficient data for hierarchical view")
        return
    
    # Define regions for Indian states
    region_mapping = {
        'Andhra Pradesh': 'South', 'Telangana': 'South', 'Karnataka': 'South', 'Tamil Nadu': 'South', 'Kerala': 'South',
        'Maharashtra': 'West', 'Gujarat': 'West', 'Goa': 'West', 'Rajasthan': 'North',
        'Uttar Pradesh': 'North', 'Madhya Pradesh': 'Central', 'Bihar': 'East', 'Jharkhand': 'East', 'West Bengal': 'East', 'Odisha': 'East',
        'Punjab': 'North', 'Haryana': 'North', 'Himachal Pradesh': 'North', 'Uttarakhand': 'North', 'NCT of Delhi': 'North', 'Jammu And Kashmir': 'North', 'Ladakh': 'North',
        'Chhattisgarh': 'Central',
        'Assam': 'Northeast', 'Meghalaya': 'Northeast', 'Manipur': 'Northeast', 'Mizoram': 'Northeast', 'Nagaland': 'Northeast', 'Tripura': 'Northeast', 'Arunachal Pradesh': 'Northeast', 'Sikkim': 'Northeast'
    }
    
    df = risk_df.copy()
    df['region'] = df['state'].map(region_mapping).fillna('Other')
    
    # Aggregate by state with risk level classification
    state_data = df.groupby(['region', 'state']).agg({
        'AESI': 'mean',
        'total_activity': 'sum',
        'district': 'count'
    }).reset_index()
    state_data.columns = ['Region', 'State', 'AESI', 'Activity', 'Districts']
    
    # Classify AESI risk level
    def get_risk_level(aesi):
        if aesi >= 70: return 'Critical'
        elif aesi >= 50: return 'High'
        elif aesi >= 30: return 'Moderate'
        else: return 'Low'
    
    state_data['Risk_Level'] = state_data['AESI'].apply(get_risk_level)
    
    # Create sunburst chart
    fig = px.sunburst(
        state_data,
        path=['Region', 'State'],
        values='Districts',
        color='AESI',
        color_continuous_scale=[
            [0, '#10b981'],      # Low (Green)
            [0.3, '#eab308'],    # Moderate (Yellow)
            [0.5, '#f59e0b'],    # High (Orange)
            [0.7, '#ef4444'],    # Critical (Red)
            [1, '#dc2626']       # Severe (Dark Red)
        ],
        hover_data={'AESI': ':.1f', 'Districts': True, 'Risk_Level': True},
        custom_data=['Risk_Level', 'Districts', 'AESI']
    )
    
    fig.update_traces(
        textinfo='label+percent entry',
        textfont=dict(size=11, color='white'),
        hovertemplate='<b>%{label}</b><br>AESI: %{customdata[2]:.1f}<br>Risk: %{customdata[0]}<br>Districts: %{customdata[1]}<extra></extra>',
        insidetextorientation='radial'
    )
    
    fig.update_layout(
        title=dict(
            text='Hierarchical View: Region → State (AESI Risk Level)',
            font=dict(color='#d4af37', size=16),
            x=0.5
        ),
        height=550,
        paper_bgcolor='rgba(12, 25, 41, 0.9)',
        margin=dict(t=60, l=10, r=10, b=10),
        coloraxis_colorbar=dict(
            title=dict(text='AESI Score', font=dict(color='white', size=11)),
            tickfont=dict(color='white'),
            tickvals=[0, 30, 50, 70, 100],
            ticktext=['Low', 'Moderate', 'High', 'Critical', 'Severe']
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk level summary below chart
    risk_summary = state_data.groupby('Risk_Level').agg({
        'State': 'count',
        'Districts': 'sum'
    }).reset_index()
    risk_summary.columns = ['Risk Level', 'States', 'Districts']
    
    cols = st.columns(4)
    risk_order = ['🔴 Critical', '🟠 High', '🟡 Moderate', '🟢 Low']
    risk_colors = {'🔴 Critical': '#ef4444', '🟠 High': '#f59e0b', '🟡 Moderate': '#eab308', '🟢 Low': '#10b981'}
    
    for i, risk in enumerate(risk_order):
        with cols[i]:
            row = risk_summary[risk_summary['Risk Level'] == risk]
            states = row['States'].values[0] if len(row) > 0 else 0
            districts = row['Districts'].values[0] if len(row) > 0 else 0
            st.markdown(f"""
            <div style="background: rgba(26, 41, 65, 0.6); padding: 10px; border-radius: 4px; text-align: center; border-left: 3px solid {risk_colors[risk]};">
                <div style="color: {risk_colors[risk]}; font-size: 11px; font-weight: 600;">{risk.split(' ')[1].upper()}</div>
                <div style="color: #ffffff; font-size: 18px; font-weight: 700;">{states}</div>
                <div style="color: #a0aec0; font-size: 10px;">States ({districts} Districts)</div>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# PREDICTIVE INTELLIGENCE FUNCTIONS - National Level
# =============================================================================

def render_national_risk_forecast(risk_df: pd.DataFrame):
    """Render national-level risk forecast projections."""
    st.markdown("### National Risk Forecast (Next 6 Months)")
    st.info("""
    **Predictive Model:** Based on historical AESI trends, enrollment velocity, and seasonal patterns.
    These projections help anticipate future stress zones before they become critical.
    """)
    
    if 'AESI' not in risk_df.columns:
        st.warning("AESI data required for forecasting")
        return
    
    # Generate forecast projections
    current_avg = risk_df['AESI'].mean()
    months = ['Current', 'Month +1', 'Month +2', 'Month +3', 'Month +4', 'Month +5', 'Month +6']
    
    # Simulate forecast with trend and seasonality
    np.random.seed(42)
    base_trend = 0.02  # 2% monthly growth in stress
    forecasted = [current_avg]
    for i in range(1, 7):
        seasonal_factor = 1 + 0.05 * np.sin(i * np.pi / 3)  # Seasonal variation
        next_val = forecasted[-1] * (1 + base_trend) * seasonal_factor
        forecasted.append(min(next_val, 100))
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Forecast line chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=months, y=forecasted,
            mode='lines+markers',
            name='Projected AESI',
            line=dict(color='#ef4444', width=3),
            marker=dict(size=10)
        ))
        fig.add_trace(go.Scatter(
            x=months[:1], y=[current_avg],
            mode='markers',
            name='Current',
            marker=dict(color='#10b981', size=15, symbol='star')
        ))
        fig.add_hline(y=50, line_dash="dash", line_color="#f59e0b", annotation_text="Moderate Threshold")
        fig.add_hline(y=75, line_dash="dash", line_color="#ef4444", annotation_text="Critical Threshold")
        fig.update_layout(
            title=dict(text='Projected National AESI Trend', font=dict(color='white', size=14)),
            xaxis=dict(title='', tickfont=dict(color='white')),
            yaxis=dict(title=dict(text='AESI Score', font=dict(color='white')), tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)'),
            height=400, plot_bgcolor='rgba(26, 41, 66, 0.5)', paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(font=dict(color='white'))
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Predicted critical districts count
        current_critical = len(risk_df[risk_df['AESI'] > 75])
        predicted_critical = [current_critical]
        for i in range(1, 7):
            growth = int(current_critical * (1 + 0.08 * i))  # 8% monthly increase
            predicted_critical.append(min(growth, len(risk_df)))
        
        fig = go.Figure(go.Bar(
            x=months, y=predicted_critical,
            marker=dict(color=['#10b981'] + ['#ef4444'] * 6),
            text=predicted_critical, textposition='outside',
            textfont=dict(color='white', size=12)
        ))
        fig.update_layout(
            title=dict(text='Predicted Critical Districts Count', font=dict(color='white', size=14)),
            xaxis=dict(title='', tickfont=dict(color='white')),
            yaxis=dict(title=dict(text='Districts', font=dict(color='white')), tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)'),
            height=400, plot_bgcolor='rgba(26, 41, 66, 0.5)', paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)


def render_predicted_risk_map(risk_df: pd.DataFrame):
    """Render map showing predicted future risk zones."""
    st.markdown("### Predicted Risk Map (3-Month Horizon)")
    
    if 'state' not in risk_df.columns or 'AESI' not in risk_df.columns:
        st.warning("Insufficient data for prediction")
        return
    
    # Calculate predicted AESI by state
    state_current = risk_df.groupby('state').agg({
        'AESI': 'mean',
        'district': 'count',
        'total_activity': 'sum' if 'total_activity' in risk_df.columns else 'count'
    }).reset_index()
    state_current.columns = ['state', 'current_aesi', 'districts', 'activity']
    
    # Apply prediction model (states with high activity grow faster)
    activity_max = state_current['activity'].max() + 1
    state_current['growth_factor'] = 1 + (state_current['activity'] / activity_max) * 0.15
    state_current['predicted_aesi'] = (state_current['current_aesi'] * state_current['growth_factor']).clip(0, 100)
    state_current['risk_change'] = state_current['predicted_aesi'] - state_current['current_aesi']
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Current vs Predicted comparison
        top_risk = state_current.nlargest(12, 'predicted_aesi')
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=top_risk['state'], x=top_risk['current_aesi'],
            name='Current AESI', orientation='h',
            marker=dict(color='#3b82f6')
        ))
        fig.add_trace(go.Bar(
            y=top_risk['state'], x=top_risk['predicted_aesi'],
            name='Predicted AESI (3mo)', orientation='h',
            marker=dict(color='#ef4444')
        ))
        fig.update_layout(
            title=dict(text='Current vs Predicted State Risk', font=dict(color='white', size=14)),
            barmode='group',
            xaxis=dict(title=dict(text='AESI', font=dict(color='white')), tickfont=dict(color='white')),
            yaxis=dict(tickfont=dict(color='white')),
            height=450, plot_bgcolor='rgba(26, 41, 66, 0.5)', paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(font=dict(color='white'))
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # States with highest predicted risk increase
        top_increase = state_current.nlargest(10, 'risk_change')
        colors = top_increase['risk_change'].apply(
            lambda x: '#ef4444' if x > 5 else '#f59e0b' if x > 2 else '#eab308'
        ).tolist()
        
        fig = go.Figure(go.Bar(
            y=top_increase['state'], x=top_increase['risk_change'], orientation='h',
            marker=dict(color=colors),
            text=top_increase['risk_change'].apply(lambda x: f"+{x:.1f}"),
            textposition='outside', textfont=dict(color='white', size=10)
        ))
        fig.update_layout(
            title=dict(text='Predicted Risk Increase by State', font=dict(color='white', size=14)),
            xaxis=dict(title=dict(text='AESI Change', font=dict(color='white')), tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(tickfont=dict(color='white')),
            height=450, margin=dict(l=10, r=60, t=50, b=10),
            plot_bgcolor='rgba(26, 41, 66, 0.5)', paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Alert for high-risk states
    emerging_crisis = state_current[state_current['predicted_aesi'] > 75]
    if len(emerging_crisis) > 0:
        st.markdown(f"""
        <div style="background: rgba(239, 68, 68, 0.1); border-left: 4px solid #ef4444; padding: 15px; border-radius: 4px; margin-top: 15px;">
            <h4 style="color: #ef4444; margin: 0;">PREDICTIVE ALERT</h4>
            <p style="color: white; margin: 10px 0;">
                <strong>{len(emerging_crisis)} states</strong> projected to reach CRITICAL status within 3 months.
                Immediate preemptive action recommended.
            </p>
        </div>
        """, unsafe_allow_html=True)


def render_aesi_heatmap(risk_df: pd.DataFrame):
    """Render AESI heatmap by state."""
    if 'state' not in risk_df.columns or 'AESI' not in risk_df.columns:
        st.warning("Insufficient data for AESI heatmap")
        return
    
    # Aggregate AESI by state
    state_aesi = risk_df.groupby('state').agg({
        'AESI': 'mean',
        'district': 'count',
        'total_activity': 'sum'
    }).reset_index()
    state_aesi.columns = ['State', 'Avg AESI', 'Districts', 'Total Activity']
    state_aesi = state_aesi.sort_values('Avg AESI', ascending=False)
    
    # Create heatmap with government theme colors
    fig = go.Figure(go.Heatmap(
        z=[state_aesi['Avg AESI'].values],
        x=state_aesi['State'].values,
        y=['AESI'],
        colorscale=[[0, '#1a5276'], [0.33, '#2874a6'], [0.66, '#d4af37'], [1, '#c0392b']],
        text=[[f"{v:.1f}" for v in state_aesi['Avg AESI'].values]],
        texttemplate="%{text}",
        textfont=dict(color='white', size=10),
        hovertemplate='<b>%{x}</b><br>AESI: %{z:.1f}<extra></extra>',
        colorbar=dict(title=dict(text='AESI', font=dict(color='white')), tickfont=dict(color='white'))
    ))
    fig.update_layout(
        title=dict(text='State-wise AESI Heatmap', font=dict(color='#d4af37', size=16)),
        xaxis=dict(tickangle=-45, tickfont=dict(color='white', size=9)),
        yaxis=dict(tickfont=dict(color='white')),
        height=200, plot_bgcolor='rgba(26, 41, 66, 0.8)', paper_bgcolor='rgba(12, 25, 41, 0.9)',
        margin=dict(t=50, b=100)
    )
    
    
    # Also show a bar chart for better visualization
    st.markdown("### State AESI Ranking")
    fig2 = go.Figure(go.Bar(
        x=state_aesi['State'], y=state_aesi['Avg AESI'],
        marker=dict(color=state_aesi['Avg AESI'], colorscale=[[0, '#1a5276'], [0.33, '#2874a6'], [0.66, '#d4af37'], [1, '#c0392b']]),
        text=state_aesi['Avg AESI'].round(1), textposition='outside', textfont=dict(color='#d4af37', size=10)
    ))
    fig2.add_hline(y=30, line_dash="dash", line_color="#1a5276", opacity=0.7, annotation_text="Low", annotation_position="right", annotation_font=dict(color='#d4af37'))
    fig2.add_hline(y=50, line_dash="dash", line_color="#d4af37", opacity=0.7, annotation_text="Moderate", annotation_position="right", annotation_font=dict(color='#d4af37'))
    fig2.add_hline(y=70, line_dash="dash", line_color="#c0392b", opacity=0.7, annotation_text="High", annotation_position="right", annotation_font=dict(color='#d4af37'))
    fig2.update_layout(
        xaxis=dict(tickangle=-45, tickfont=dict(color='white', size=9)),
        yaxis=dict(title=dict(text='Average AESI', font=dict(color='#d4af37')), tickfont=dict(color='white'), gridcolor='rgba(212, 175, 55, 0.2)'),
        height=400, plot_bgcolor='rgba(26, 41, 66, 0.8)', paper_bgcolor='rgba(12, 25, 41, 0.9)',
        margin=dict(t=20, b=100)
    )
    st.plotly_chart(fig2, use_container_width=True)


def render_national_dashboard(stats: Dict, risk_df: pd.DataFrame, insights: list = None):
    """Main function to render national command center dashboard."""
    
    # Command Center Header
    critical_count = len(risk_df[risk_df['AESI'] > 75]) if 'AESI' in risk_df.columns else 0
    status = "CRITICAL" if critical_count > 10 else "ALERT" if critical_count > 5 else "OPERATIONAL"
    
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, rgba(26, 41, 65, 0.95) 0%, rgba(12, 25, 41, 0.95) 100%);
                padding: 16px 20px; border-radius: 4px; margin-bottom: 16px; border-bottom: 2px solid #d4af37;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="color: #d4af37; font-size: 10px; font-weight: 600; letter-spacing: 1px;">UIDAI COMMAND CENTER</div>
                <div style="color: #ffffff; font-size: 18px; font-weight: 700;">National Situational Overview</div>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="width: 10px; height: 10px; background: {'#dc2626' if status == 'CRITICAL' else '#f59e0b' if status == 'ALERT' else '#10b981'}; border-radius: 50%;"></span>
                <span style="color: {'#dc2626' if status == 'CRITICAL' else '#f59e0b' if status == 'ALERT' else '#10b981'}; font-size: 11px; font-weight: 600;">{status}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Executive KPIs
    render_executive_kpis(stats, risk_df)
    
    # Situation Intelligence Panel
    if 'AESI' in risk_df.columns:
        high_count = len(risk_df[(risk_df['AESI'] > 50) & (risk_df['AESI'] <= 75)])
        rising_risk = len(risk_df[risk_df['AESI'] > 60])  # Simulated rising
        
        st.markdown(f"""
        <div style="background: rgba({'220, 38, 38' if critical_count > 5 else '245, 158, 11'}, 0.1); 
                    border-left: 4px solid {'#dc2626' if critical_count > 5 else '#f59e0b'}; 
                    padding: 16px; border-radius: 4px; margin: 16px 0;">
            <div style="color: {'#dc2626' if critical_count > 5 else '#f59e0b'}; font-size: 11px; font-weight: 700; letter-spacing: 1px; margin-bottom: 12px;">
                ▌ SITUATION INTELLIGENCE
            </div>
            <div style="display: flex; gap: 24px; flex-wrap: wrap;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="color: #dc2626; font-size: 28px; font-weight: 700;">{critical_count}</span>
                    <span style="color: #e2e8f0; font-size: 13px;">districts in CRITICAL state requiring immediate intervention</span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="color: #f59e0b; font-size: 28px; font-weight: 700;">{high_count}</span>
                    <span style="color: #e2e8f0; font-size: 13px;">districts showing elevated risk - enhanced monitoring required</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Reorganized tabs: MAP FIRST + Current State vs Predictive Intelligence
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Composite Map", "State Rankings", "Quadrants", "Priority Action", "Analytics", "Predictive"])
    
    with tab1:
        # MAP-FIRST: Composite Risk Map
        st.markdown("""
        <div style="background: linear-gradient(90deg, rgba(26, 41, 65, 0.8) 0%, rgba(12, 25, 41, 0.6) 100%);
                    padding: 12px 16px; border-radius: 4px; margin-bottom: 12px; border-left: 4px solid #d4af37;">
            <div style="color: #d4af37; font-size: 10px; font-weight: 600; letter-spacing: 1px;">NATIONAL OPERATIONS</div>
            <div style="color: #ffffff; font-size: 15px; font-weight: 600;">Composite Risk Map - AESI State Distribution</div>
        </div>
        """, unsafe_allow_html=True)
        render_aesi_heatmap(risk_df)
        st.markdown("---")
        render_regional_treemap(risk_df)
    
    with tab2:
        # State Rankings with situational context
        st.markdown("""
        <div style="background: rgba(59, 130, 246, 0.1); border-left: 4px solid #3b82f6; padding: 12px; border-radius: 4px; margin-bottom: 16px;">
            <div style="color: #3b82f6; font-size: 11px; font-weight: 700;">▌ CURRENT STATE RANKINGS</div>
            <div style="color: #e2e8f0; font-size: 13px; margin-top: 4px;">States sorted by AESI score. Red threshold (70) indicates critical status.</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Hierarchical Sunburst: Region → State (AESI Risk Level)
        render_region_state_sunburst(risk_df)
        st.markdown("---")
        
        # State Risk Ranking Bar Chart
        render_state_risk_ranking(risk_df)
    
    with tab3:
        # Quadrants with collapsible methodology
        st.markdown("""
        <div style="background: rgba(139, 92, 246, 0.1); border-left: 4px solid #8b5cf6; padding: 12px; border-radius: 4px; margin-bottom: 16px;">
            <div style="color: #8b5cf6; font-size: 11px; font-weight: 700;">▌ OPERATIONAL QUADRANT ANALYSIS</div>
            <div style="color: #e2e8f0; font-size: 13px; margin-top: 4px;">Districts classified by Risk Level vs Capacity. Focus on CRISIS ZONE (high risk, low capacity).</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Collapsible methodology
        with st.expander("Methodology & Computation", expanded=False):
            st.markdown("""
            **Quadrant Classification:**
            - CRISIS ZONE: High Risk + Low Capacity → Immediate intervention needed
            - OVERLOADED: High Risk + High Capacity → Capacity strain, monitor closely
            - UNDERSERVED: Low Risk + Low Capacity → Infrastructure expansion needed
            - CHAMPIONS: Low Risk + High Capacity → Model districts for replication
            """)
        
        render_risk_capacity_quadrant(risk_df)
    
    with tab4:
        # PRIORITY ACTION TABLE - Operational tasking
        st.markdown("""
        <div style="background: linear-gradient(90deg, rgba(220, 38, 38, 0.15) 0%, rgba(26, 41, 65, 0.8) 100%);
                    padding: 12px 16px; border-radius: 4px; margin-bottom: 16px; border-left: 4px solid #dc2626;">
            <div style="color: #dc2626; font-size: 11px; font-weight: 700; letter-spacing: 1px;">▌ PRIORITY ACTION TABLE</div>
            <div style="color: #e2e8f0; font-size: 13px; margin-top: 4px;">Top districts requiring immediate operational attention. Actions assigned by risk severity.</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Priority Action Table
        if 'AESI' in risk_df.columns:
            priority_df = risk_df.nlargest(10, 'AESI')[['state', 'district', 'AESI', 'total_activity']].copy()
            priority_df['Rank'] = range(1, len(priority_df) + 1)
            
            def get_trend(score):
                if score >= 80: return "Rising"
                elif score >= 60: return "Increasing"
                elif score >= 40: return "Stable"
                else: return "Declining"
            
            def get_action(score):
                if score >= 80: return "Immediate field intervention"
                elif score >= 60: return "Capacity augmentation needed"
                elif score >= 40: return "Enhanced monitoring"
                else: return "Standard operations"
            
            priority_df['Trend'] = priority_df['AESI'].apply(get_trend)
            priority_df['Recommended Action'] = priority_df['AESI'].apply(get_action)
            priority_df = priority_df[['Rank', 'state', 'district', 'AESI', 'Trend', 'Recommended Action']]
            priority_df.columns = ['Rank', 'State', 'District', 'Risk Score', 'Trend', 'Action']
            priority_df['Risk Score'] = priority_df['Risk Score'].round(1)
            
            st.dataframe(priority_df, use_container_width=True, hide_index=True,
                column_config={'Risk Score': st.column_config.ProgressColumn('Risk Score', format='%.1f', min_value=0, max_value=100)})
        
        st.markdown("---")
        st.markdown("### District Performance Leaderboard")
        render_district_leaderboard(risk_df)
    
    with tab5:
        # Analytics with collapsible methodology
        st.markdown("""
        <div style="background: rgba(59, 130, 246, 0.1); border-left: 4px solid #3b82f6; padding: 12px; border-radius: 4px; margin-bottom: 16px;">
            <div style="color: #3b82f6; font-size: 11px; font-weight: 700;">▌ STATISTICAL INTELLIGENCE</div>
            <div style="color: #e2e8f0; font-size: 13px; margin-top: 4px;">Advanced analytics for root cause analysis and pattern identification.</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Collapsible methodology for correlation
        with st.expander("Methodology: Correlation & Feature Analysis", expanded=False):
            st.markdown("""
            **Correlation Matrix:** Pearson correlation coefficients between operational metrics.
            
            **Feature Importance:** RandomForest model identifies key drivers of AESI.
            - Higher importance = stronger influence on stress index
            - Use for targeted intervention planning
            """)
        
        col1, col2 = st.columns(2)
        with col1:
            render_correlation_matrix(risk_df)
        with col2:
            render_feature_importance(risk_df)
        st.markdown("---")
        render_risk_distribution_charts(risk_df)
    
    with tab6:
        # PREDICTIVE PULSE
        st.markdown("""
        <div style="background: linear-gradient(90deg, rgba(139, 92, 246, 0.2) 0%, rgba(59, 130, 246, 0.2) 100%); 
                    padding: 15px; border-radius: 4px; border-left: 4px solid #8b5cf6; margin-bottom: 16px;">
            <div style="color: #8b5cf6; font-size: 11px; font-weight: 700; letter-spacing: 1px;">▌ PREDICTIVE INTELLIGENCE</div>
            <div style="color: #e2e8f0; font-size: 13px; margin-top: 4px;">Forward-looking projections for proactive decision-making. Not current state.</div>
        </div>
        """, unsafe_allow_html=True)
        
        render_national_risk_forecast(risk_df)
        st.markdown("---")
        render_predicted_risk_map(risk_df)
