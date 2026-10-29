import streamlit as st

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.dashboard import render_dashboard

render_dashboard()
