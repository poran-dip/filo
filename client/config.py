import streamlit as st

BACKEND_URL = st.secrets.get("BACKEND_URL", "http://localhost:8000")
