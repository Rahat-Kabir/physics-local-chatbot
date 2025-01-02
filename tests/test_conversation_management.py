import pytest
import streamlit as st
from app import start_new_conversation  # Assuming start_new_conversation is in app.py

class TestConversationManagement:
    def test_new_conversation(self, mock_st_session):
        """Test creating new conversation"""
        start_new_conversation()
        assert 'conversations' in st.session_state
        assert 'active_conversation' in st.session_state
        assert len(st.session_state.conversations) == 1

    def test_conversation_history(self, mock_st_session):
        """Test conversation history management"""
        start_new_conversation()
        conv_id = st.session_state.active_conversation
        assert st.session_state.conversations[conv_id]['history'] == []
