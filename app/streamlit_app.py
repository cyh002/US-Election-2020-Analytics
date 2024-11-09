# streamlit_app.py
# How to run: 
# python -m streamlit run app/streamlit_app.py

import streamlit as st
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
        - **Daily Analysis:** Perform in-depth analysis on the day's data, including anomaly detection, key topics, and key trends.
        - **Choropleth Map:** Visualize comparative sentiment scores between hashtags by state.
        - **Time Series Analysis:** Track sentiment trends over time for selected hashtags.
        - **Word Clouds:** Explore common words used in tweets for each hashtag.
        - **Sentiment Distribution:** Understand the distribution of sentiment scores by hashtag.
        - **User Influence:** Analyze the relationship between user followers and sentiment.
        
        Use the sidebar to navigate through different sections of the dashboard.
        """)

if __name__ == "__main__":
    app = TwitterSentimentApp()
    
