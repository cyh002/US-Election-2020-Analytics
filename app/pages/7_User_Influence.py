import streamlit as st
from general_utils.app_state import init_state
import pandas as pd
import plotly.express as px
from app.general_utils.streamlit_filters import StreamlitFilters

class UserInfluencePage:
    def __init__(self):
        init_state()
        self.data = st.session_state['data']
        self.filters = StreamlitFilters(self.data)
        self.selected_hashtag = []
        self.filtered_data = pd.DataFrame()
        self.main()

    def sidebar_filters(self):
        st.sidebar.header("Filters")
        with st.sidebar.expander("📊 Filter Options"):
            # Hashtag Filter
            hashtags = self.data['hashtag'].unique().tolist()
            st.markdown("**Select Hashtag(s):**")
            for hashtag in hashtags:
                if st.sidebar.checkbox(hashtag, value=True):
                    self.selected_hashtag.append(hashtag)

    def apply_filters(self):
        self.filtered_data = self.data[self.data['hashtag'].isin(self.selected_hashtag)]

    def user_influence(self):
        st.header("📈 User Influence Analysis")

        if self.filtered_data.empty:
            st.warning("No data available to display the user influence analysis.")
            return

        fig = px.scatter(
            self.filtered_data,
            x='user_followers_count',
            y='normalized_score',
            size='engagement',
            color='hashtag',
            color_discrete_map={
                'trump': 'red',
                'biden': 'blue',
                'both': 'green'
            },
            title="User Followers vs. Normalized Sentiment Scores",
            labels={
                'user_followers_count': 'User Followers Count',
                'normalized_score': 'Normalized Sentiment Score',
                'engagement': 'Engagement'
            },
            log_x=True,
            hover_data=['state', 'hashtag']
        )

        fig.update_layout(
            legend_title_text='Hashtag',
            margin={"r": 0, "t": 30, "l": 0, "b": 0}
        )

        st.plotly_chart(fig, use_container_width=True)

    def main(self):
        self.filters.sidebar_filters()
        self.filters.apply_filters()
        self.selected_hashtag = self.filters.selected_hashtag
        self.filtered_data = self.filters.filtered_data
        
        self.user_influence()

if __name__ == "__main__":
    UserInfluencePage()
