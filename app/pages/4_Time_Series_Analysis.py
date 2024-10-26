# pages/4_Time_Series_Analysis.py

import streamlit as st
from general_utils.app_state import init_state
import pandas as pd
import plotly.express as px
from general_utils.streamlit_filters import StreamlitFilters

class TimeSeriesAnalysisPage:
    def __init__(self):
        init_state()
        self.data = st.session_state['data']
        self.filtered_data = pd.DataFrame()
        self.filters = StreamlitFilters(self.data)
        self.main()

    def time_series_analysis(self):
        st.header("📈 Time Series Analysis")

        # Group by date and hashtag to calculate mean normalized scores
        time_series = self.filtered_data.groupby(['created_date', 'hashtag'])['normalized_score'].mean().reset_index()

        # Define color mapping for hashtags
        color_map = {
            'trump': 'red',
            'biden': 'blue',
            'both': 'green'
        }

        # Create the line plot
        fig = px.line(
            time_series,
            x='created_date',
            y='normalized_score',
            color='hashtag',
            color_discrete_map=color_map,
            title=f"Average Normalized Scores Over Time in {self.selected_state}",
            labels={
                'created_date': 'Date',
                'normalized_score': 'Avg Normalized Score',
                'hashtag': 'Hashtag'
            }
        )

        fig.update_layout(
            legend_title_text='Hashtag',
            margin={"r": 0, "t": 30, "l": 0, "b": 0}
        )

        st.plotly_chart(fig, use_container_width=True)

    def main(self):
        self.filters.sidebar_filters()
        self.filters.apply_filters()
        self.filtered_data = self.filters.filtered_data
        self.selected_state = self.filters.selected_state
        self.time_series_analysis()
        
if __name__ == "__main__":
    TimeSeriesAnalysisPage()
