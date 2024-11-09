# streamlit_app.py
# How to run: 
# python -m streamlit run app/streamlit_app.py

import streamlit as st
from pathlib import Path
import subprocess
import sys
import os

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from app.general_utils.data_loader import DataLoader
from app.general_utils.app_state import init_state

class TwitterSentimentApp:
    def __init__(self):
        self.setup_page()
        init_state()
        self.load_shared_data()
        self.introduction()

    def setup_page(self):
        st.set_page_config(
            page_title="Twitter Sentiment Analysis",
            page_icon=":bar_chart:",
            layout="wide"
        )

    def load_shared_data(self):
        self.data = st.session_state['data']
        self.data_loader = st.session_state['data_loader']
        self.llm_analyzer = st.session_state['llm_analyzer']
        self.geojson_state_names = st.session_state['geojson_state_names']

    def introduction(self):
        st.markdown("""
        # Twitter Sentiment Analysis Dashboard

        Welcome to the Twitter Sentiment Analysis Dashboard! This application visualizes the sentiment of tweets across different US states and hashtags. Explore the data through interactive maps, time series plots, word clouds, and more.

        **Features:**
        - **Data Overview:** View the data and its structure.
        - **User Analysis:** Analyze the relationship between user followers and sentiment.
        - **Daily Analysis:** Perform in-depth analysis on the day's data, including anomaly detection, key topics, and key trends, using LLM models.
        - **Choropleth Map:** Visualize comparative sentiment scores between hashtags by state.
        - **Sentiment Distribution:** Understand the distribution of sentiment scores by hashtag.
        - **Time Series Analysis:** Track sentiment trends over time for selected hashtags.
        - **Word Clouds:** Explore common words used in tweets for each hashtag.
        
        Use the sidebar to navigate through different sections of the dashboard.
        """)
        
        st.markdown("""
        📚 [View Full Glossary and Metrics](/Glossary)
        """)

        
        # Authors section with columns and custom styling
        st.markdown("## Meet the Team")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### Christopher Chi Yang Hoo
            [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/christopher-chi-yang-hoo-570698bb/)
            Data Scientist & ML Engineer
            """)
        
        with col2:
            st.markdown("""
            ### Victoria Kwong Jia Ying
            [![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:kwong.victoriaa@gmail.com)
            Data Scientist & ML Engineer
            """)
        
        with col3:
            st.markdown("""
            ### Anthony Kwa
            [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/anthony-kwa/)
            Data Scientist & ML Engineer
            """)



if __name__ == "__main__":
    app = TwitterSentimentApp()
    
    
    
