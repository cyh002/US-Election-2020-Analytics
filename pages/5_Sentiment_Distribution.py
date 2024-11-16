# pages/3_Sentiment_Distribution.py

import streamlit as st
from general_utils.app_state import init_state
import pandas as pd
import plotly.express as px
from general_utils.streamlit_filters import StreamlitFilters

class SentimentDistributionPage:
    def __init__(self):
        init_state()
        self.data = st.session_state['data']
        self.selected_hashtag = []
        self.filtered_data = pd.DataFrame()
        self.filters = StreamlitFilters(self.data)

        self.main()
       
    def sentiment_distribution_by_hashtag(self):
        st.header("📊 Sentiment Distribution by Hashtag")

        if self.filtered_data.empty:
            st.warning("No data available to display the sentiment distribution.")
            return

        hashtags = self.filtered_data['hashtag'].unique()
        for hashtag in hashtags:
            st.markdown(f"#### #{hashtag.capitalize()}")
            hashtag_data = self.filtered_data[self.filtered_data['hashtag'] == hashtag]
            fig = px.histogram(
                hashtag_data,
                x='normalized_score',
                nbins=50,
                title=f"Distribution of Normalized Sentiment Scores for #{hashtag}",
                labels={'normalized_score': 'Normalized Sentiment Score'},
                color_discrete_sequence=['#636EFA']
            )
            st.plotly_chart(fig, use_container_width=True)

    def main(self):
        self.filters.sidebar_filters()
        self.filters.apply_filters()
        self.filtered_data = self.filters.filtered_data
        self.sentiment_distribution_by_hashtag()

if __name__ == "__main__":
    SentimentDistributionPage()
