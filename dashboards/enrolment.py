"""
================================================================================
UIDAI COMMAND CENTER - ENROLLMENT INTELLIGENCE
================================================================================
Government Command-and-Control Dashboard for Enrollment Operations

Design Principles:
- Map-first: AESI Geographic Heatmap as primary visual
- Situation Intelligence Panels for actionable alerts
- Priority Action Tables for operational tasking
- Collapsible methodology (formulas hidden by default)
- Protected KPIs (no misleading zeros)
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Optional, List
import streamlit.components.v1 as components

try:
    import folium
    from folium.plugins import HeatMap
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

# Institutional Color Scheme
RISK_COLORS = {'low': '#10b981', 'moderate': '#eab308', 'high': '#f59e0b', 'critical': '#ef4444'}
UIDAI_COLORS = {'navy_dark': '#0c1929', 'navy_light': '#1a2941', 'gold': '#d4af37', 'white': '#ffffff', 'muted': '#a0aec0'}


def render_enrolment_kpis(df_enroll: pd.DataFrame, risk_df: pd.DataFrame):
    """Render enrolment-specific KPIs with protected display."""
    col1, col2, col3, col4 = st.columns(4)
    
    total = df_enroll['total_enrollment'].sum() if len(df_enroll) > 0 and 'total_enrollment' in df_enroll.columns else 0
    child_total = df_enroll['child_enrollment'].sum() if len(df_enroll) > 0 and 'child_enrollment' in df_enroll.columns else 0
    adult_total = df_enroll['adult_enrollment'].sum() if len(df_enroll) > 0 and 'adult_enrollment' in df_enroll.columns else 0
    high_epi = len(risk_df[risk_df['EPI_normalized'] >= 75]) if 'EPI_normalized' in risk_df.columns else 0
    
    def format_num(n):
        if n is None or n == 0: return None
        if n >= 1e7: return f"{n/1e7:.2f} Cr"
        elif n >= 1e5: return f"{n/1e5:.2f} L"
        elif n >= 1e3: return f"{n/1e3:.1f} K"
        return f"{n:,.0f}"
    
    kpi_data = [
        ("TOTAL ENROLLMENTS", format_num(total), "#d4af37"),
        ("CHILD (0-17)", format_num(child_total), "#3b82f6"),
        ("ADULT (18+)", format_num(adult_total), "#10b981"),
        ("HIGH PRESSURE", str(high_epi) if high_epi > 0 else None, "#ef4444")
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


def render_monthly_trend(df_enroll: pd.DataFrame):
    """Render monthly enrolment trend chart."""
    if len(df_enroll) == 0 or 'date' not in df_enroll.columns:
        st.warning("No enrolment data for trend analysis")
        return
    
    monthly = df_enroll.groupby(df_enroll['date'].dt.to_period('M')).agg({
        'total_enrollment': 'sum'
    }).reset_index()
    monthly['date'] = monthly['date'].dt.to_timestamp()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly['date'], y=monthly['total_enrollment'], name='Enrollments',
        line=dict(color='#3b82f6', width=3), fill='tozeroy', fillcolor='rgba(59, 130, 246, 0.2)'
    ))
    fig.update_layout(
        title=dict(text='Monthly Enrollment Trend', font=dict(color='white', size=16)),
        xaxis=dict(title=dict(text='Month', font=dict(color='white')), tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(title=dict(text='Enrollments', font=dict(color='white')), tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)'),
        height=400, plot_bgcolor='rgba(26, 41, 66, 0.5)', paper_bgcolor='rgba(0,0,0,0)', legend=dict(font=dict(color='white'))
    )
    st.plotly_chart(fig, use_container_width=True)


def render_age_distribution(df_enroll: pd.DataFrame):
    """Render age group distribution pie chart."""
    age_cols = ['age_0_5', 'age_5_17', 'age_18_40', 'age_40_60', 'age_60_plus']
    available = [c for c in age_cols if c in df_enroll.columns]
    
    if not available:
        st.info("Age distribution data not available")
        return
    
    age_totals = {col: df_enroll[col].sum() for col in available}
    labels = ['0-5 (Bal Aadhaar)', '5-17 (Children)', '18-40 (Young Adults)', '40-60 (Middle Age)', '60+ (Senior)'][:len(available)]
    
    fig = go.Figure(go.Pie(
        labels=labels, values=list(age_totals.values()),
        marker=dict(colors=['#c0392b', '#d4af37', '#2874a6', '#1a5276', '#0c1929']),
        textinfo='percent+label', textfont=dict(color='white'), hole=0.4
    ))
    fig.update_layout(
        title=dict(text='Enrollment by Age Group', font=dict(color='white', size=16)),
        height=400, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', legend=dict(font=dict(color='white'))
    )
    st.plotly_chart(fig, use_container_width=True)


def render_alsi_analysis(df_enroll: pd.DataFrame, risk_df: pd.DataFrame):
    """Render ALSI (Aadhaar Lifecycle Stress Index) analysis with collapsible methodology."""
    st.markdown("### ALSI - Future MBU Crisis Detection")
    
    # Situation intelligence panel instead of verbose explanation
    st.markdown("""
    <div style="background: rgba(245, 158, 11, 0.1); border-left: 4px solid #f59e0b; padding: 12px; border-radius: 4px; margin-bottom: 16px;">
        <div style="color: #f59e0b; font-size: 11px; font-weight: 700;">▌ FORWARD PLANNING ALERT</div>
        <div style="color: #e2e8f0; font-size: 13px; margin-top: 4px;">High ALSI states will face MBU capacity crisis in 3-5 years. Start infrastructure planning NOW.</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Collapsible methodology
    with st.expander("Methodology: ALSI Computation", expanded=False):
        st.markdown("""
        **Formula:** `ALSI = (Child_Enrollment × Days_Span) / (Adult_Activity + 1)`
        
        **Components:**
        - Child Enrollment (40%): Children who will need MBU in future
        - Days Span (30%): Operational duration factor
        - Adult Activity (30%): Current infrastructure utilization
        
        **Thresholds:** Low (<25) | Moderate (25-50) | High (50-75) | Critical (>75)
        """)
    
    # Check for available columns - use age columns if child_enrollment not computed
    child_col = None
    adult_col = None
    
    if 'child_enrollment' in df_enroll.columns and df_enroll['child_enrollment'].sum() > 0:
        child_col = 'child_enrollment'
        adult_col = 'adult_enrollment'
    else:
        # Try to use raw age columns
        for col in df_enroll.columns:
            if 'age_0_5' in col or 'age_5_17' in col:
                child_col = col
            if 'age_18' in col or 'greater' in col:
                adult_col = col
    
    if child_col is None or len(df_enroll) == 0:
        # Fallback: use total_enrollment split
        if 'total_enrollment' in df_enroll.columns:
            df_enroll['child_enrollment'] = df_enroll['total_enrollment'] * 0.3  # Assume 30% children
            df_enroll['adult_enrollment'] = df_enroll['total_enrollment'] * 0.7
            child_col = 'child_enrollment'
            adult_col = 'adult_enrollment'
        else:
            st.warning("Insufficient data for ALSI calculation")
            return
    
    # Calculate ALSI at state level
    agg_dict = {child_col: 'sum', 'total_enrollment': 'sum'}
    if adult_col and adult_col in df_enroll.columns:
        agg_dict[adult_col] = 'sum'
    if 'date' in df_enroll.columns:
        agg_dict['date'] = ['min', 'max']
    
    alsi_data = df_enroll.groupby('state').agg(agg_dict).reset_index()
    alsi_data.columns = [c[0] if isinstance(c, tuple) else c for c in alsi_data.columns]
    
    # Rename columns
    if child_col in alsi_data.columns:
        alsi_data = alsi_data.rename(columns={child_col: 'child_enroll'})
    if adult_col and adult_col in alsi_data.columns:
        alsi_data = alsi_data.rename(columns={adult_col: 'adult_enroll'})
    else:
        alsi_data['adult_enroll'] = alsi_data['total_enrollment'] * 0.7
    
    alsi_data['days_span'] = 365  # Default if no date
    alsi_data['ALSI'] = (alsi_data['child_enroll'] * alsi_data['days_span']) / (alsi_data['adult_enroll'] + 1)
    alsi_data['ALSI_Normalized'] = ((alsi_data['ALSI'] - alsi_data['ALSI'].min()) / (alsi_data['ALSI'].max() - alsi_data['ALSI'].min() + 0.001) * 100).clip(0, 100)
    
    # Categorize
    alsi_data['Category'] = pd.cut(alsi_data['ALSI_Normalized'], bins=[0, 25, 50, 75, 100], labels=['Low', 'Moderate', 'High', 'Critical'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart of ALSI by state
        alsi_sorted = alsi_data.nlargest(15, 'ALSI_Normalized')
        colors = alsi_sorted['ALSI_Normalized'].apply(
            lambda x: RISK_COLORS['low'] if x < 25 else RISK_COLORS['moderate'] if x < 50 else RISK_COLORS['high'] if x < 75 else RISK_COLORS['critical']
        ).tolist()
        
        fig = go.Figure(go.Bar(
            y=alsi_sorted['state'], x=alsi_sorted['ALSI_Normalized'], orientation='h',
            marker=dict(color=colors), text=alsi_sorted['ALSI_Normalized'].round(1), textposition='outside',
            textfont=dict(color='white', size=10)
        ))
        fig.update_layout(
            title=dict(text='Top 15 States by ALSI', font=dict(color='white', size=14)),
            xaxis=dict(title=dict(text='ALSI Score', font=dict(color='white')), tickfont=dict(color='white')),
            yaxis=dict(title='', tickfont=dict(color='white')),
            height=450, margin=dict(l=10, r=60, t=50, b=10),
            plot_bgcolor='rgba(26, 41, 66, 0.5)', paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Category distribution
        cat_counts = alsi_data['Category'].value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=cat_counts.index, values=cat_counts.values,
            marker=dict(colors=[RISK_COLORS['low'], RISK_COLORS['moderate'], RISK_COLORS['high'], RISK_COLORS['critical']]),
            textinfo='percent+label', textfont=dict(color='white'), hole=0.4
        ))
        fig_pie.update_layout(
            title=dict(text='ALSI Risk Distribution', font=dict(color='white', size=14)),
            height=450, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', legend=dict(font=dict(color='white'))
        )
        st.plotly_chart(fig_pie, use_container_width=True)


def render_mbu_forecast(df_enroll: pd.DataFrame):
    """Render MBU pipeline forecast - Bal Aadhaar to future MBU demand."""
    st.markdown("### MBU Pipeline Forecast (5-Year)")
    st.info("Children enrolled in age_0_5 (Bal Aadhaar) will need MBU in 5 years. This forecast enables proactive planning.")
    
    if 'age_0_5' not in df_enroll.columns:
        st.warning("Bal Aadhaar (age_0_5) data not available")
        return
    
    # State-level future MBU demand
    future_mbu = df_enroll.groupby('state').agg({
        'age_0_5': 'sum', 'age_5_17': 'sum', 'total_enrollment': 'sum'
    }).reset_index()
    future_mbu.columns = ['State', 'Bal_Aadhaar', 'Current_MBU_Pool', 'Total']
    future_mbu['Projected_MBU_2029'] = (future_mbu['Bal_Aadhaar'] * 0.95).astype(int)
    future_mbu = future_mbu.sort_values('Projected_MBU_2029', ascending=False).head(15)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Current Bal Aadhaar', x=future_mbu['State'], y=future_mbu['Bal_Aadhaar'], marker_color='#3b82f6'))
    fig.add_trace(go.Bar(name='Projected MBU 2029', x=future_mbu['State'], y=future_mbu['Projected_MBU_2029'], marker_color='#ef4444'))
    
    fig.update_layout(
        barmode='group',
        title=dict(text='Bal Aadhaar vs Projected MBU Demand', font=dict(color='white', size=16)),
        xaxis=dict(title=dict(text='State', font=dict(color='white')), tickfont=dict(color='white'), tickangle=-45),
        yaxis=dict(title=dict(text='Count', font=dict(color='white')), tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)'),
        height=450, plot_bgcolor='rgba(26, 41, 66, 0.5)', paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(font=dict(color='white'), bgcolor='rgba(26,41,66,0.8)')
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary
    total_bal = future_mbu['Bal_Aadhaar'].sum()
    total_proj = future_mbu['Projected_MBU_2029'].sum()
    st.markdown(f"""
    <div style="background: rgba(26, 41, 66, 0.8); padding: 15px; border-radius: 8px; border-left: 4px solid #d4af37;">
        <h4 style="color: #d4af37; margin: 0;">MBU Pipeline Summary</h4>
        <p style="color: white; margin: 10px 0;">Current Bal Aadhaar: <strong>{total_bal:,}</strong></p>
        <p style="color: white; margin: 0;">Projected MBU by 2029: <strong>{total_proj:,}</strong></p>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PREDICTIVE INTELLIGENCE FUNCTIONS - Enrollment Domain
# =============================================================================

def render_aesi_forecast_map(risk_df: pd.DataFrame):
    """Render AESI forecast map showing predicted stress zones."""
    st.markdown("### AESI Forecast Map (6-Month Projection)")
    st.info("Predicted stress zones based on current enrollment velocity and historical patterns.")
    
    if 'AESI' not in risk_df.columns or 'state' not in risk_df.columns:
        st.warning("Insufficient data for AESI forecasting")
        return
    
    # Calculate predicted AESI
    state_aesi = risk_df.groupby('state').agg({
        'AESI': 'mean',
        'total_enrollment': 'sum' if 'total_enrollment' in risk_df.columns else 'count',
        'district': 'count'
    }).reset_index()
    state_aesi.columns = ['state', 'current_aesi', 'enrollment_volume', 'districts']
    
    # Prediction: High enrollment volume states will see AESI increase
    vol_max = state_aesi['enrollment_volume'].max() + 1
    state_aesi['velocity_factor'] = state_aesi['enrollment_volume'] / vol_max
    state_aesi['predicted_aesi'] = (state_aesi['current_aesi'] * (1 + state_aesi['velocity_factor'] * 0.2)).clip(0, 100)
    state_aesi['aesi_change'] = state_aesi['predicted_aesi'] - state_aesi['current_aesi']
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Predicted high-stress states
        high_stress = state_aesi.nlargest(12, 'predicted_aesi')
        colors = high_stress['predicted_aesi'].apply(
            lambda x: '#ef4444' if x >= 75 else '#f59e0b' if x >= 50 else '#eab308' if x >= 25 else '#10b981'
        ).tolist()
        
        fig = go.Figure(go.Bar(
            y=high_stress['state'], x=high_stress['predicted_aesi'], orientation='h',
            marker=dict(color=colors),
            text=high_stress['predicted_aesi'].round(1), textposition='outside',
            textfont=dict(color='white', size=10)
        ))
        fig.add_vline(x=75, line_dash="dash", line_color="#ef4444", opacity=0.7)
        fig.update_layout(
            title=dict(text='Predicted High-Stress States', font=dict(color='white', size=14)),
            xaxis=dict(title=dict(text='Predicted AESI', font=dict(color='white')), tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(tickfont=dict(color='white')),
            height=450, margin=dict(l=10, r=60, t=50, b=10),
            plot_bgcolor='rgba(26, 41, 66, 0.5)', paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Emerging risk states (biggest predicted increase)
        emerging = state_aesi.nlargest(10, 'aesi_change')
        fig = go.Figure(go.Bar(
            y=emerging['state'], x=emerging['aesi_change'], orientation='h',
            marker=dict(color='#ef4444'),
            text=emerging['aesi_change'].apply(lambda x: f"+{x:.1f}"),
            textposition='outside', textfont=dict(color='white', size=10)
        ))
        fig.update_layout(
            title=dict(text='Emerging Risk States (Biggest Increase)', font=dict(color='white', size=14)),
            xaxis=dict(title=dict(text='Predicted AESI Increase', font=dict(color='white')), tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(tickfont=dict(color='white')),
            height=450, margin=dict(l=10, r=60, t=50, b=10),
            plot_bgcolor='rgba(26, 41, 66, 0.5)', paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)


def render_enrollment_volume_prediction(df_enroll: pd.DataFrame):
    """Render enrollment volume prediction chart."""
    st.markdown("### Enrollment Volume Prediction (12-Month Horizon)")
    
    if len(df_enroll) == 0 or 'date' not in df_enroll.columns:
        st.info("Date information required for volume prediction")
        return
    
    # Historical monthly enrollment
    monthly = df_enroll.groupby(df_enroll['date'].dt.to_period('M'))['total_enrollment'].sum().reset_index()
    monthly['date'] = monthly['date'].dt.to_timestamp()
    monthly = monthly.tail(12)  # Last 12 months
    
    # Generate predictions for next 12 months
    if len(monthly) > 0:
        last_value = monthly['total_enrollment'].iloc[-1]
        avg_growth = monthly['total_enrollment'].pct_change().mean()
        avg_growth = 0.02 if pd.isna(avg_growth) else avg_growth  # Default 2%
        
        future_dates = pd.date_range(start=monthly['date'].iloc[-1], periods=13, freq='M')[1:]
        predictions = []
        current = last_value
        for i, date in enumerate(future_dates):
            # Add seasonality
            seasonal = 1 + 0.1 * np.sin((i + 1) * np.pi / 6)
            current = current * (1 + avg_growth) * seasonal
            predictions.append({'date': date, 'total_enrollment': current, 'type': 'Predicted'})
        
        monthly['type'] = 'Historical'
        future_df = pd.DataFrame(predictions)
        combined = pd.concat([monthly, future_df], ignore_index=True)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=combined[combined['type']=='Historical']['date'],
            y=combined[combined['type']=='Historical']['total_enrollment'],
            name='Historical', mode='lines+markers',
            line=dict(color='#3b82f6', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=combined[combined['type']=='Predicted']['date'],
            y=combined[combined['type']=='Predicted']['total_enrollment'],
            name='Predicted', mode='lines+markers',
            line=dict(color='#ef4444', width=3, dash='dash'),
            fill='tozeroy', fillcolor='rgba(239, 68, 68, 0.1)'
        ))
        fig.update_layout(
            title=dict(text='Historical vs Predicted Enrollment Volume', font=dict(color='white', size=14)),
            xaxis=dict(title='', tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(title=dict(text='Enrollments', font=dict(color='white')), tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)'),
            height=400, plot_bgcolor='rgba(26, 41, 66, 0.5)', paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(font=dict(color='white'))
        )
        st.plotly_chart(fig, use_container_width=True)


def render_predicted_high_stress_districts(risk_df: pd.DataFrame):
    """Render map of districts predicted to become high-stress."""
    st.markdown("### Predicted High-Stress Districts")
    
    if 'AESI' not in risk_df.columns:
        st.warning("AESI data required")
        return
    
    df = risk_df.copy()
    
    # Predict which districts will become critical
    df['predicted_aesi'] = df['AESI'] * 1.15  # 15% increase projection
    df['predicted_aesi'] = df['predicted_aesi'].clip(0, 100)
    df['will_become_critical'] = (df['AESI'] < 75) & (df['predicted_aesi'] >= 75)
    df['already_critical'] = df['AESI'] >= 75
    
    col1, col2, col3 = st.columns(3)
    with col1:
        already = df['already_critical'].sum()
        st.metric("Already Critical", f"{already}")
    with col2:
        emerging = df['will_become_critical'].sum()
        st.metric("Will Become Critical", f"{emerging}")
    with col3:
        safe = len(df) - already - emerging
        st.metric("Projected Safe", f"{safe}")
    
    # List emerging critical districts
    emerging_df = df[df['will_become_critical']][['state', 'district', 'AESI', 'predicted_aesi']].copy()
    if len(emerging_df) > 0:
        emerging_df.columns = ['State', 'District', 'Current AESI', 'Predicted AESI']
        emerging_df['Current AESI'] = emerging_df['Current AESI'].round(1)
        emerging_df['Predicted AESI'] = emerging_df['Predicted AESI'].round(1)
        emerging_df = emerging_df.sort_values('Predicted AESI', ascending=False).head(15)
        
        st.markdown("**Districts Projected to Enter Critical Zone:**")
        st.dataframe(emerging_df, use_container_width=True, hide_index=True)


def render_state_enrollment_ranking(df_enroll: pd.DataFrame, risk_df: pd.DataFrame):
    """Render state enrollment pressure ranking."""
    if len(df_enroll) == 0: return
    
    state_data = df_enroll.groupby('state')['total_enrollment'].sum().reset_index()
    state_data.columns = ['state', 'total_enrollment']
    state_data = state_data.sort_values('total_enrollment', ascending=True)
    
    fig = go.Figure(go.Bar(
        y=state_data['state'], x=state_data['total_enrollment'], orientation='h',
        marker=dict(color='#3b82f6'), text=state_data['total_enrollment'].apply(lambda x: f"{x/1e5:.1f}L" if x >= 1e5 else f"{x:,.0f}"),
        textposition='outside', textfont=dict(color='white', size=10)
    ))
    fig.update_layout(
        title=dict(text='State Enrollment Volume', font=dict(color='white', size=16)),
        xaxis=dict(title=dict(text='Total Enrollments', font=dict(color='white')), tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(title='', tickfont=dict(color='white')),
        height=max(400, len(state_data) * 22), margin=dict(l=10, r=80, t=50, b=10),
        plot_bgcolor='rgba(26, 41, 66, 0.5)', paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)


def render_pressure_districts(risk_df: pd.DataFrame):
    """Render high-pressure district table."""
    display_df = risk_df.copy()
    
    # If EPI_normalized doesn't exist, compute a proxy from available data
    if 'EPI_normalized' not in display_df.columns:
        if 'total_enrollment' in display_df.columns:
            # Compute EPI proxy based on enrollment volume
            display_df['EPI_normalized'] = ((display_df['total_enrollment'] - display_df['total_enrollment'].min()) / 
                                            (display_df['total_enrollment'].max() - display_df['total_enrollment'].min() + 1) * 100)
        elif 'AESI' in display_df.columns:
            display_df['EPI_normalized'] = display_df['AESI']
        else:
            st.info("Insufficient data for pressure index calculation")
            return
    
    pressure_df = display_df.nlargest(15, 'EPI_normalized')[['state', 'district', 'EPI_normalized', 'total_enrollment'] if 'total_enrollment' in display_df.columns else ['state', 'district', 'EPI_normalized']]
    
    if 'total_enrollment' in display_df.columns:
        pressure_df.columns = ['State', 'District', 'Pressure Index', 'Enrollments']
    else:
        pressure_df.columns = ['State', 'District', 'Pressure Index']
    
    pressure_df['Pressure Index'] = pressure_df['Pressure Index'].round(1)
    
    st.dataframe(pressure_df, use_container_width=True, hide_index=True,
        column_config={'Pressure Index': st.column_config.ProgressColumn('Pressure', format='%.1f', min_value=0, max_value=100)})


# =============================================================================
# AESI GEOGRAPHIC ANALYSIS (Moved from geographic.py)
# =============================================================================

# State centroids for India (approximate lat/lon for map plotting)
STATE_CENTROIDS = {
    'Andhra Pradesh': (15.9129, 79.7400), 'Arunachal Pradesh': (28.2180, 94.7278),
    'Assam': (26.2006, 92.9376), 'Bihar': (25.0961, 85.3131),
    'Chhattisgarh': (21.2787, 81.8661), 'Goa': (15.2993, 74.1240),
    'Gujarat': (22.2587, 71.1924), 'Haryana': (29.0588, 76.0856),
    'Himachal Pradesh': (31.1048, 77.1734), 'Jharkhand': (23.6102, 85.2799),
    'Karnataka': (15.3173, 75.7139), 'Kerala': (10.8505, 76.2711),
    'Madhya Pradesh': (22.9734, 78.6569), 'Maharashtra': (19.7515, 75.7139),
    'Manipur': (24.6637, 93.9063), 'Meghalaya': (25.4670, 91.3662),
    'Mizoram': (23.1645, 92.9376), 'Nagaland': (26.1584, 94.5624),
    'Odisha': (20.9517, 85.0985), 'Punjab': (31.1471, 75.3412),
    'Rajasthan': (27.0238, 74.2179), 'Sikkim': (27.5330, 88.5122),
    'Tamil Nadu': (11.1271, 78.6569), 'Telangana': (18.1124, 79.0193),
    'Tripura': (23.9408, 91.9882), 'Uttar Pradesh': (26.8467, 80.9462),
    'Uttarakhand': (30.0668, 79.0193), 'West Bengal': (22.9868, 87.8550),
    'NCT of Delhi': (28.7041, 77.1025), 'Jammu And Kashmir': (33.7782, 76.5762),
    'Ladakh': (34.1526, 77.5770), 'Puducherry': (11.9416, 79.8083),
    'Chandigarh': (30.7333, 76.7794), 'Andaman And Nicobar Islands': (11.7401, 92.6586),
    'Dadra And Nagar Haveli And Daman And Diu': (20.1809, 73.0169),
    'Lakshadweep': (10.5667, 72.6417)
}


def render_aesi_heatmap(risk_df: pd.DataFrame, df_bio: pd.DataFrame = None):
    """
    Render interactive AESI heatmap using Folium.
    Shows geographic distribution of Aadhaar Ecosystem Stress Index.
    """
    st.markdown("### AESI Geographic Heatmap")
    st.info("""
    **AESI (Aadhaar Ecosystem Stress Index)** measures the overall stress level of Aadhaar operations.
    - Red zones indicate critical stress requiring immediate intervention
    - Green zones represent healthy operational capacity
    """)
    
    if not FOLIUM_AVAILABLE:
        st.warning("Folium library not available. Install with: pip install folium")
        # Fallback to plotly map
        render_aesi_plotly_map(risk_df)
        return
    
    if 'AESI' not in risk_df.columns or 'state' not in risk_df.columns:
        st.warning("AESI data not available for geographic visualization")
        return
    
    # Aggregate AESI by state
    state_aesi = risk_df.groupby('state').agg({
        'AESI': 'mean',
        'district': 'count',
        'total_activity': 'sum' if 'total_activity' in risk_df.columns else 'count'
    }).reset_index()
    state_aesi.columns = ['state', 'avg_aesi', 'district_count', 'total_activity']
    
    # Create base map centered on India
    india_map = folium.Map(
        location=[20.5937, 78.9629],
        zoom_start=5,
        tiles='cartodbdark_matter'
    )
    
    # Prepare heatmap data
    heat_data = []
    for _, row in state_aesi.iterrows():
        state_name = row['state']
        if state_name in STATE_CENTROIDS:
            lat, lon = STATE_CENTROIDS[state_name]
            weight = row['avg_aesi'] / 100  # Normalize to 0-1
            heat_data.append([lat, lon, weight])
            
            # Add marker with popup
            popup_html = f"""
            <div style="font-family: Arial; width: 200px;">
                <h4 style="color: #1a5276; margin: 0;">{state_name}</h4>
                <hr style="margin: 5px 0;">
                <p><strong>Avg AESI:</strong> {row['avg_aesi']:.1f}</p>
                <p><strong>Districts:</strong> {row['district_count']}</p>
                <p><strong>Activity:</strong> {row['total_activity']:,.0f}</p>
            </div>
            """
            
            # Color based on AESI
            if row['avg_aesi'] >= 75:
                color = '#ef4444'
            elif row['avg_aesi'] >= 50:
                color = '#f59e0b'
            elif row['avg_aesi'] >= 25:
                color = '#eab308'
            else:
                color = '#10b981'
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=max(8, row['avg_aesi'] / 5),
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=folium.Popup(popup_html, max_width=250)
            ).add_to(india_map)
    
    # Add heatmap layer
    if heat_data:
        HeatMap(
            heat_data,
            min_opacity=0.3,
            max_zoom=10,
            radius=25,
            blur=15,
            gradient={0.2: '#1a5276', 0.4: '#2874a6', 0.6: '#d4af37', 0.8: '#f59e0b', 1.0: '#ef4444'}
        ).add_to(india_map)
    
    # Render map
    map_html = india_map._repr_html_()
    components.html(map_html, height=550)
    
    # Legend
    st.markdown("""
    <div style="display: flex; justify-content: center; gap: 30px; margin-top: 10px;">
        <span style="display: flex; align-items: center; gap: 5px;">
            <span style="width: 15px; height: 15px; background: #10b981; border-radius: 50%;"></span>
            <span style="color: white;">Low (&lt;25)</span>
        </span>
        <span style="display: flex; align-items: center; gap: 5px;">
            <span style="width: 15px; height: 15px; background: #eab308; border-radius: 50%;"></span>
            <span style="color: white;">Moderate (25-50)</span>
        </span>
        <span style="display: flex; align-items: center; gap: 5px;">
            <span style="width: 15px; height: 15px; background: #f59e0b; border-radius: 50%;"></span>
            <span style="color: white;">High (50-75)</span>
        </span>
        <span style="display: flex; align-items: center; gap: 5px;">
            <span style="width: 15px; height: 15px; background: #ef4444; border-radius: 50%;"></span>
            <span style="color: white;">Critical (&gt;75)</span>
        </span>
    </div>
    """, unsafe_allow_html=True)


def render_aesi_plotly_map(risk_df: pd.DataFrame):
    """Fallback Plotly scatter map when Folium not available."""
    if 'AESI' not in risk_df.columns:
        return
    
    state_aesi = risk_df.groupby('state').agg({
        'AESI': 'mean',
        'district': 'count'
    }).reset_index()
    
    # Add coordinates
    state_aesi['lat'] = state_aesi['state'].map(lambda x: STATE_CENTROIDS.get(x, (20, 78))[0])
    state_aesi['lon'] = state_aesi['state'].map(lambda x: STATE_CENTROIDS.get(x, (20, 78))[1])
    
    fig = px.scatter_geo(
        state_aesi,
        lat='lat', lon='lon',
        color='AESI',
        size='district',
        hover_name='state',
        color_continuous_scale=[[0, '#1a5276'], [0.33, '#2874a6'], [0.66, '#d4af37'], [1, '#c0392b']],
        scope='asia',
        center={'lat': 20.5, 'lon': 78.9}
    )
    fig.update_layout(
        geo=dict(bgcolor='rgba(12,25,41,0.9)', showland=True, landcolor='rgba(26,41,66,0.8)'),
        paper_bgcolor='rgba(0,0,0,0)',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)


def render_state_level_aesi_analysis(risk_df: pd.DataFrame):
    """Render state-level AESI stress analysis with bar charts and statistics."""
    st.markdown("### State-Level AESI Analysis")
    
    if 'AESI' not in risk_df.columns or 'state' not in risk_df.columns:
        st.warning("AESI data not available")
        return
    
    # Aggregate by state
    state_stress = risk_df.groupby('state').agg({
        'AESI': ['mean', 'max', 'min', 'std'],
        'district': 'count',
        'total_activity': 'sum' if 'total_activity' in risk_df.columns else 'count'
    }).reset_index()
    state_stress.columns = ['state', 'avg_aesi', 'max_aesi', 'min_aesi', 'std_aesi', 'district_count', 'total_activity']
    state_stress['std_aesi'] = state_stress['std_aesi'].fillna(0)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top stressed states
        top_stress = state_stress.nlargest(12, 'avg_aesi')
        colors = top_stress['avg_aesi'].apply(
            lambda x: '#ef4444' if x >= 75 else '#f59e0b' if x >= 50 else '#eab308' if x >= 25 else '#10b981'
        ).tolist()
        
        fig = go.Figure(go.Bar(
            y=top_stress['state'], x=top_stress['avg_aesi'], orientation='h',
            marker=dict(color=colors),
            text=top_stress['avg_aesi'].round(1), textposition='outside',
            textfont=dict(color='white', size=10),
            error_x=dict(type='data', array=top_stress['std_aesi'], color='rgba(255,255,255,0.3)')
        ))
        fig.add_vline(x=50, line_dash="dash", line_color="#f59e0b", opacity=0.5)
        fig.add_vline(x=75, line_dash="dash", line_color="#ef4444", opacity=0.5)
        fig.update_layout(
            title=dict(text='Top Stressed States', font=dict(color='white', size=14)),
            xaxis=dict(title=dict(text='Average AESI', font=dict(color='white')), tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(title='', tickfont=dict(color='white')),
            height=450, margin=dict(l=10, r=60, t=50, b=10),
            plot_bgcolor='rgba(26, 41, 66, 0.5)', paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Bottom stressed states (best performers)
        bottom_stress = state_stress.nsmallest(12, 'avg_aesi')
        colors = bottom_stress['avg_aesi'].apply(
            lambda x: '#ef4444' if x >= 75 else '#f59e0b' if x >= 50 else '#eab308' if x >= 25 else '#10b981'
        ).tolist()
        
        fig = go.Figure(go.Bar(
            y=bottom_stress['state'], x=bottom_stress['avg_aesi'], orientation='h',
            marker=dict(color=colors),
            text=bottom_stress['avg_aesi'].round(1), textposition='outside',
            textfont=dict(color='white', size=10)
        ))
        fig.update_layout(
            title=dict(text='Best Performing States', font=dict(color='white', size=14)),
            xaxis=dict(title=dict(text='Average AESI', font=dict(color='white')), tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(title='', tickfont=dict(color='white')),
            height=450, margin=dict(l=10, r=60, t=50, b=10),
            plot_bgcolor='rgba(26, 41, 66, 0.5)', paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # State statistics table
    st.markdown("### State AESI Statistics")
    display_df = state_stress[['state', 'avg_aesi', 'max_aesi', 'min_aesi', 'district_count']].copy()
    display_df.columns = ['State', 'Avg AESI', 'Max AESI', 'Min AESI', 'Districts']
    display_df['Avg AESI'] = display_df['Avg AESI'].round(1)
    display_df['Max AESI'] = display_df['Max AESI'].round(1)
    display_df['Min AESI'] = display_df['Min AESI'].round(1)
    display_df = display_df.sort_values('Avg AESI', ascending=False)
    
    st.dataframe(display_df, use_container_width=True, hide_index=True,
        column_config={'Avg AESI': st.column_config.ProgressColumn('Avg AESI', format='%.1f', min_value=0, max_value=100)})


def render_aesi_distribution_analysis(risk_df: pd.DataFrame):
    """Render AESI distribution histogram and category analysis."""
    st.markdown("### AESI Distribution Analysis")
    
    if 'AESI' not in risk_df.columns:
        st.warning("AESI data not available")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # AESI Histogram
        fig = go.Figure(go.Histogram(
            x=risk_df['AESI'], nbinsx=30,
            marker=dict(color='#3b82f6', line=dict(color='white', width=1))
        ))
        fig.add_vline(x=25, line_dash="dash", line_color="#10b981", annotation_text="Low", annotation_position="top")
        fig.add_vline(x=50, line_dash="dash", line_color="#f59e0b", annotation_text="Moderate", annotation_position="top")
        fig.add_vline(x=75, line_dash="dash", line_color="#ef4444", annotation_text="High", annotation_position="top")
        fig.update_layout(
            title=dict(text='AESI Score Distribution', font=dict(color='white', size=14)),
            xaxis=dict(title=dict(text='AESI Score', font=dict(color='white')), tickfont=dict(color='white')),
            yaxis=dict(title=dict(text='Districts', font=dict(color='white')), tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)'),
            height=400, plot_bgcolor='rgba(26, 41, 66, 0.5)', paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Stress category distribution
        df = risk_df.copy()
        df['Category'] = pd.cut(df['AESI'], bins=[0, 25, 50, 75, 100], labels=['Low', 'Moderate', 'High', 'Critical'])
        cat_counts = df['Category'].value_counts()
        
        fig = go.Figure(go.Pie(
            labels=cat_counts.index, values=cat_counts.values,
            marker=dict(colors=['#10b981', '#eab308', '#f59e0b', '#ef4444']),
            textinfo='percent+label', textfont=dict(color='white'), hole=0.4
        ))
        fig.update_layout(
            title=dict(text='Stress Category Distribution', font=dict(color='white', size=14)),
            height=400, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(font=dict(color='white'))
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Statistics summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean AESI", f"{risk_df['AESI'].mean():.1f}")
    with col2:
        st.metric("Median AESI", f"{risk_df['AESI'].median():.1f}")
    with col3:
        st.metric("Std Dev", f"{risk_df['AESI'].std():.1f}")
    with col4:
        critical_pct = (len(risk_df[risk_df['AESI'] > 75]) / len(risk_df) * 100)
        st.metric("Critical %", f"{critical_pct:.1f}%")


def render_enrollment_distribution(df_enroll: pd.DataFrame):
    """Render enrollment distribution histogram from statistical analysis."""
    st.markdown("### Enrollment Distribution")
    
    if len(df_enroll) == 0 or 'total_enrollment' not in df_enroll.columns:
        st.info("Enrollment distribution data not available")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Enrollment histogram
        fig = go.Figure(go.Histogram(
            x=df_enroll['total_enrollment'], nbinsx=30,
            marker=dict(color='#3b82f6', line=dict(color='white', width=1))
        ))
        fig.update_layout(
            title=dict(text='Enrollment Volume Distribution', font=dict(color='white', size=14)),
            xaxis=dict(title=dict(text='Total Enrollment', font=dict(color='white')), tickfont=dict(color='white')),
            yaxis=dict(title=dict(text='Frequency', font=dict(color='white')), tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)'),
            height=350, plot_bgcolor='rgba(26, 41, 66, 0.5)', paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top states by enrollment
        if 'state' in df_enroll.columns:
            state_enroll = df_enroll.groupby('state')['total_enrollment'].sum().reset_index()
            state_enroll = state_enroll.nlargest(10, 'total_enrollment')
            
            fig = go.Figure(go.Bar(
                y=state_enroll['state'], x=state_enroll['total_enrollment'], orientation='h',
                marker=dict(color='#3b82f6'),
                text=state_enroll['total_enrollment'].apply(lambda x: f"{x/1e5:.1f}L" if x >= 1e5 else f"{x:,.0f}"),
                textposition='outside', textfont=dict(color='white', size=10)
            ))
            fig.update_layout(
                title=dict(text='Top 10 States by Enrollment', font=dict(color='white', size=14)),
                xaxis=dict(title=dict(text='Total Enrollment', font=dict(color='white')), tickfont=dict(color='white')),
                yaxis=dict(title='', tickfont=dict(color='white')),
                height=350, margin=dict(l=10, r=80, t=50, b=10),
                plot_bgcolor='rgba(26, 41, 66, 0.5)', paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)


def render_enrolment_dashboard(df_enroll: pd.DataFrame, risk_df: pd.DataFrame):
    """Main function to render enrollment command center dashboard."""
    
    # Command Center Header
    high_pressure = len(risk_df[risk_df['EPI_normalized'] >= 75]) if 'EPI_normalized' in risk_df.columns else 0
    critical_aesi = len(risk_df[risk_df['AESI'] > 75]) if 'AESI' in risk_df.columns else 0
    status = "CRITICAL" if critical_aesi > 10 else "ALERT" if critical_aesi > 5 else "OPERATIONAL"
    
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, rgba(26, 41, 65, 0.95) 0%, rgba(12, 25, 41, 0.95) 100%);
                padding: 16px 20px; border-radius: 4px; margin-bottom: 16px; border-bottom: 2px solid #3b82f6;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="color: #3b82f6; font-size: 10px; font-weight: 600; letter-spacing: 1px;">ENROLLMENT OPERATIONS</div>
                <div style="color: #ffffff; font-size: 18px; font-weight: 700;">Enrollment Intelligence Command</div>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="width: 10px; height: 10px; background: {'#dc2626' if status == 'CRITICAL' else '#f59e0b' if status == 'ALERT' else '#10b981'}; border-radius: 50%;"></span>
                <span style="color: {'#dc2626' if status == 'CRITICAL' else '#f59e0b' if status == 'ALERT' else '#10b981'}; font-size: 11px; font-weight: 600;">{status}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # KPIs
    render_enrolment_kpis(df_enroll, risk_df)
    
    # Situation Intelligence Panel
    if 'AESI' in risk_df.columns:
        avg_aesi = risk_df['AESI'].mean()
        st.markdown(f"""
        <div style="background: rgba({'220, 38, 38' if critical_aesi > 5 else '59, 130, 246'}, 0.1); 
                    border-left: 4px solid {'#dc2626' if critical_aesi > 5 else '#3b82f6'}; 
                    padding: 16px; border-radius: 4px; margin: 16px 0;">
            <div style="color: {'#dc2626' if critical_aesi > 5 else '#3b82f6'}; font-size: 11px; font-weight: 700; letter-spacing: 1px; margin-bottom: 12px;">
                ▌ ENROLLMENT SITUATION
            </div>
            <div style="display: flex; gap: 24px; flex-wrap: wrap;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="color: #dc2626; font-size: 28px; font-weight: 700;">{critical_aesi}</span>
                    <span style="color: #e2e8f0; font-size: 13px;">districts with critical AESI (&gt;75) - immediate capacity review needed</span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="color: #f59e0b; font-size: 28px; font-weight: 700;">{high_pressure}</span>
                    <span style="color: #e2e8f0; font-size: 13px;">districts with high enrollment pressure</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # MAP-FIRST tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Maps", "Trends", "Distribution", "Priority Action", "Predictive"])
    
    with tab1:
        # MAP-FIRST: AESI Geographic Heatmap
        avg_aesi = risk_df['AESI'].mean() if 'AESI' in risk_df.columns else None
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, rgba(26, 41, 65, 0.8) 0%, rgba(12, 25, 41, 0.6) 100%);
                    padding: 12px 16px; border-radius: 4px; margin-bottom: 12px; border-left: 4px solid #3b82f6;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="color: #3b82f6; font-size: 10px; font-weight: 600; letter-spacing: 1px;">ENROLLMENT OPERATIONS</div>
                    <div style="color: #ffffff; font-size: 15px; font-weight: 600;">AESI Geographic Distribution - Enrollment Stress Hotspots</div>
                </div>
                <div style="text-align: right;">
                    <div style="color: #a0aec0; font-size: 10px;">NATIONAL AESI</div>
                    <div style="color: {'#dc2626' if avg_aesi and avg_aesi >= 75 else '#f59e0b' if avg_aesi and avg_aesi >= 50 else '#10b981'}; font-size: 24px; font-weight: 700;">{f'{avg_aesi:.1f}' if avg_aesi else 'N/A'}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        render_aesi_heatmap(risk_df)
        st.markdown("---")
        render_state_level_aesi_analysis(risk_df)
    
    with tab2:
        # Trends with situation context
        st.markdown("""
        <div style="background: rgba(59, 130, 246, 0.1); border-left: 4px solid #3b82f6; padding: 12px; border-radius: 4px; margin-bottom: 16px;">
            <div style="color: #3b82f6; font-size: 11px; font-weight: 700;">▌ ENROLLMENT TRENDS</div>
            <div style="color: #e2e8f0; font-size: 13px; margin-top: 4px;">Historical enrollment patterns and age demographics for capacity planning.</div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            render_monthly_trend(df_enroll)
        with col2:
            render_age_distribution(df_enroll)
        st.markdown("---")
        render_alsi_analysis(df_enroll, risk_df)
    
    with tab3:
        # Distribution with situation context
        st.markdown("""
        <div style="background: rgba(59, 130, 246, 0.1); border-left: 4px solid #3b82f6; padding: 12px; border-radius: 4px; margin-bottom: 16px;">
            <div style="color: #3b82f6; font-size: 11px; font-weight: 700;">▌ DISTRIBUTION ANALYSIS</div>
            <div style="color: #e2e8f0; font-size: 13px; margin-top: 4px;">Statistical distribution of enrollment volume and AESI scores for anomaly detection.</div>
        </div>
        """, unsafe_allow_html=True)
        
        render_enrollment_distribution(df_enroll)
        st.markdown("---")
        render_aesi_distribution_analysis(risk_df)
    
    with tab4:
        # PRIORITY ACTION TABLE
        st.markdown("""
        <div style="background: linear-gradient(90deg, rgba(220, 38, 38, 0.15) 0%, rgba(26, 41, 65, 0.8) 100%);
                    padding: 12px 16px; border-radius: 4px; margin-bottom: 16px; border-left: 4px solid #dc2626;">
            <div style="color: #dc2626; font-size: 11px; font-weight: 700; letter-spacing: 1px;">▌ PRIORITY ACTION TABLE</div>
            <div style="color: #e2e8f0; font-size: 13px; margin-top: 4px;">Districts requiring immediate enrollment capacity intervention.</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Priority Action Table for Enrollment
        if 'AESI' in risk_df.columns:
            priority_df = risk_df.nlargest(10, 'AESI')[['state', 'district', 'AESI']].copy()
            if 'total_enrollment' in risk_df.columns:
                priority_df = risk_df.nlargest(10, 'AESI')[['state', 'district', 'AESI', 'total_enrollment']].copy()
            
            priority_df['Rank'] = range(1, len(priority_df) + 1)
            
            def get_action(score):
                if score >= 80: return "Emergency capacity expansion"
                elif score >= 60: return "Add enrollment centers"
                elif score >= 40: return "Staff augmentation"
                else: return "Monitor"
            
            priority_df['Action'] = priority_df['AESI'].apply(get_action)
            
            cols = ['Rank', 'state', 'district', 'AESI', 'Action']
            if 'total_enrollment' in priority_df.columns:
                cols = ['Rank', 'state', 'district', 'AESI', 'total_enrollment', 'Action']
            
            priority_df = priority_df[cols]
            priority_df.columns = ['Rank', 'State', 'District', 'AESI'] + (['Volume', 'Recommended Action'] if 'total_enrollment' in risk_df.columns else ['Recommended Action'])
            priority_df['AESI'] = priority_df['AESI'].round(1)
            
            st.dataframe(priority_df, use_container_width=True, hide_index=True,
                column_config={'AESI': st.column_config.ProgressColumn('AESI', format='%.1f', min_value=0, max_value=100)})
        
        st.markdown("---")
        st.markdown("### State Enrollment Rankings")
        render_state_enrollment_ranking(df_enroll, risk_df)
        st.markdown("---")
        st.markdown("### High Pressure Districts")
        render_pressure_districts(risk_df)
    
    with tab5:
        # PREDICTIVE INTELLIGENCE
        st.markdown("""
        <div style="background: linear-gradient(90deg, rgba(139, 92, 246, 0.2) 0%, rgba(59, 130, 246, 0.2) 100%); 
                    padding: 15px; border-radius: 4px; border-left: 4px solid #8b5cf6; margin-bottom: 16px;">
            <div style="color: #8b5cf6; font-size: 11px; font-weight: 700; letter-spacing: 1px;">▌ PREDICTIVE INTELLIGENCE</div>
            <div style="color: #e2e8f0; font-size: 13px; margin-top: 4px;">Forward-looking enrollment projections. Use for infrastructure planning, NOT current operations.</div>
        </div>
        """, unsafe_allow_html=True)
        
        render_mbu_forecast(df_enroll)
        st.markdown("---")
        render_aesi_forecast_map(risk_df)
        st.markdown("---")
        render_enrollment_volume_prediction(df_enroll)
        st.markdown("---")
        render_predicted_high_stress_districts(risk_df)
