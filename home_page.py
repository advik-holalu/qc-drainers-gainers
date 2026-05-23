import streamlit as st


def render_home() -> None:
    # Vertical space at top
    st.write("")
    st.write("")

    # Centered title + subtitle (text-align center, not just centered column)
    st.markdown(
        "<h1 style='text-align: center;'>QC Drainers/Gainers Dashboard</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; color: gray;'>"
        "Weekly quick-commerce market share tracking for GO DESi"
        "</p>",
        unsafe_allow_html=True,
    )

    # More vertical space
    st.write("")
    st.write("")
    st.write("")

    # Two buttons centered
    _, c2, c3, _ = st.columns([1, 1, 1, 1])
    with c2:
        if st.button("View Dashboard", use_container_width=True, key="goto_dashboard"):
            st.session_state.page = "dashboard"
            st.rerun()
    with c3:
        if st.button("Upload Data", use_container_width=True, key="goto_upload"):
            st.session_state.page = "upload"
            st.rerun()
