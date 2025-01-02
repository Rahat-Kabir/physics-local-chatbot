import pytest
import streamlit as st
from app import init_session_state  # Assuming you have this function in app.py to initialize session state
from PyPDF2 import PdfReader
import os

@pytest.fixture(scope="function")
def mock_st_session():
    # Fixture to mock Streamlit session state
    init_session_state()
    yield st.session_state

@pytest.fixture(scope="function")
def sample_pdf():
    # Fixture to load a sample PDF file
    return os.path.join(os.getcwd(), "physics9-10.pdf")
