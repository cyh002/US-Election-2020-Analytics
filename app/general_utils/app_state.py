# app_state.py
import streamlit as st
from app.general_utils.data_loader import DataLoader
from app.general_utils.llm_analyzer import LLMAnalyzer
from app.general_utils.misc_app_utils import get_geojson_state_names

def init_state():
    if 'data_loader' not in st.session_state:
        st.session_state['data_loader'] = DataLoader('conf/config.yaml')
        st.session_state['data'] = st.session_state['data_loader'].get_data()

    if 'llm_analyzer' not in st.session_state:
        data_loader = st.session_state['data_loader']
        st.session_state['llm_analyzer'] = LLMAnalyzer(
            openai_api_key=data_loader.openai_api_key,
            base_url=data_loader.openai_base_url,
            model=data_loader.openai_model
        )

    if 'geojson_state_names' not in st.session_state:
        geojson_url = st.session_state['data_loader'].geojson_url
        st.session_state['geojson_state_names'] = get_geojson_state_names(geojson_url)
