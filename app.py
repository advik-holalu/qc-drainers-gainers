import streamlit as st

from dashboard_page import render_dashboard
from home_page import render_home
from upload_page import render_upload

st.set_page_config(page_title="QC Drainers/Gainers Dashboard", layout="wide")

if "page" not in st.session_state:
    st.session_state.page = "home"

# Small "← Home" button on sub-pages (top-left)
if st.session_state.page != "home":
    home_col, _ = st.columns([1, 11])
    with home_col:
        if st.button("← Home", key="home_btn", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()

if st.session_state.page == "home":
    render_home()
elif st.session_state.page == "dashboard":
    render_dashboard()
elif st.session_state.page == "upload":
    render_upload()
