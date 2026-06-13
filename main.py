# main.py
import streamlit as st
from frontend import render_ui

# Configure the page settings
st.set_page_config(
    page_title="Pauna Medical Assistant",
    page_icon="🤖",
    layout="centered"
)

# Run the app
if __name__ == "__main__":

    render_ui()