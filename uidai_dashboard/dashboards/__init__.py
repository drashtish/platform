"""
================================================================================
UIDAI GOVERNANCE INTELLIGENCE DASHBOARD - DASHBOARD MODULE EXPORTS
================================================================================
Dashboard Structure (Reorganized):
- National Overview: Executive KPIs, Correlation, Quadrants, Leaderboard
- Enrollment Intelligence: Map View (AESI Heatmap), Trend View, Distribution View, Ranking View
- Biometric Intelligence: Map View, Trend View (BUSI), Distribution View, Ranking View, Anomaly Detection
- Demographic Intelligence: Map View, Trend View (AIS), Distribution View, Ranking View, Dual Threat
- Policy Intelligence: Priority Actions, Cost-Benefit, Scenario Analysis
================================================================================
"""

from .national import render_national_dashboard
from .enrolment import render_enrolment_dashboard
from .biometric import render_biometric_dashboard
from .updates import render_demographic_dashboard

__all__ = [
    'render_national_dashboard',
    'render_enrolment_dashboard',
    'render_biometric_dashboard',
    'render_demographic_dashboard'
]
