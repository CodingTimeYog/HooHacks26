import streamlit as st
import os
import time

def wait_for_data():
    cache = "data/processed/cache.json"
    if not os.path.exists(cache):
        st.markdown("""
        <div style="text-align:center; padding-top:4rem;">
            <h1>🌾 foreGASt</h1>
            <p>Running initial data refresh — this takes a few minutes on first start...</p>
        </div>
        """, unsafe_allow_html=True)
        with st.spinner("Waiting for data pipeline..."):
            while not os.path.exists(cache):
                time.sleep(3)
                st.rerun()

wait_for_data()
st.switch_page("pages/login.py")