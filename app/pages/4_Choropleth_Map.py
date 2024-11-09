import streamlit as st
from app.general_utils.app_state import init_state
import plotly.express as px
import requests
from datetime import date
import pandas as pd
from typing import Tuple, List
from app.general_utils.streamlit_filters import StreamlitFilters  # Make sure to import your StreamlitFilters class

class ChoroplethMapPage:
    def __init__(self) -> None:
        """
        Initializes the ChoroplethMapPage class, setting up state, loading data, and initializing filters.
        """
        init_state()
        self.data = st.session_state['data']
        self.data_loader = st.session_state['data_loader']
        self.geojson_url = self.data_loader.geojson_url
        self.geojson_state_names = st.session_state['geojson_state_names']

        # Initialize StreamlitFilters instance for managing sidebar filters and filtered data
        self.filters = StreamlitFilters(self.data)
        
        self.main()

    def create_choropleth_map(
        self, df: pd.DataFrame, geojson_url: str, geojson_states: List[str], date_range: Tuple[date, date]
    ) -> None:
        """
        Creates and displays a choropleth map showing comparative sentiment scores between 'biden' and 'trump' by state.
        """
        st.subheader(
            f"🌎 Comparative Sentiment Scores by State\n🕒 From {date_range[0]} to {date_range[1]}"
        )

        # Clean state names
        df['state'] = df['state'].str.strip().str.title()

        # Calculate average normalized scores and total number of tweets per state and hashtag
        state_hashtag_scores = df.groupby(['state', 'hashtag']).agg(
            normalized_score=('normalized_score', 'mean'),
            total_tweets=('tweet_id', 'size'),
            engagement=('engagement', 'mean')
        ).reset_index()

        # Pivot the DataFrame to have 'trump' and 'biden' as separate columns
        pivot_df = state_hashtag_scores.pivot(index='state', columns='hashtag').reset_index()

        # Flatten MultiIndex columns
        pivot_df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in pivot_df.columns.values]

        # Ensure both 'trump' and 'biden' columns are present
        for suffix in ['normalized_score', 'total_tweets', 'engagement']:
            for hashtag in ['trump', 'biden']:
                column_name = f"{suffix}_{hashtag}"
                if column_name not in pivot_df.columns:
                    if suffix == 'normalized_score' or suffix == 'engagement':
                        pivot_df[column_name] = 0.0
                    elif suffix == 'total_tweets':
                        pivot_df[column_name] = 0

        # Calculate comparative score (Biden - Trump)
        pivot_df['comparative_score'] = pivot_df['normalized_score_biden'] - pivot_df['normalized_score_trump']

        # Calculate total tweets across both hashtags
        pivot_df['total_tweets'] = pivot_df['total_tweets_biden'] + pivot_df['total_tweets_trump']

        # Calculate average engagement across both hashtags
        pivot_df['average_engagement'] = (pivot_df['engagement_biden'] + pivot_df['engagement_trump']) / 2

        # Exclude states not present in GeoJSON
        pivot_df = pivot_df[pivot_df['state'].isin(geojson_states)]

        # Select and rename columns for clarity
        display_df = pivot_df[[
            'state',
            'normalized_score_biden',
            'normalized_score_trump',
            'comparative_score',
            'average_engagement',
            'total_tweets'
        ]].rename(columns={
            'normalized_score_biden': 'Biden Normalized Score',
            'normalized_score_trump': 'Trump Normalized Score',
            'average_engagement': 'Average Engagement',
            'comparative_score': 'Comparative Score (Biden - Trump)',
            'total_tweets': 'Total Tweets'
        })

        # Display the DataFrame
        st.write("**Comparative Scores DataFrame:**")
        st.dataframe(display_df)

        if pivot_df.empty:
            st.warning("No data available to display the choropleth map.")
            return

        # Fetch GeoJSON data
        try:
            response = requests.get(geojson_url)
            response.raise_for_status()
            us_states_geojson = response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching GeoJSON data: {e}")
            return

        # Determine min and max of 'comparative_score' for symmetric color scaling
        min_score = pivot_df['comparative_score'].min()
        max_score = pivot_df['comparative_score'].max()
        abs_max = max(abs(min_score), abs(max_score))

        # Define custom color scale: negative red, zero white, positive blue
        custom_color_scale = [
            (0.0, "red"),
            (0.5, "white"),
            (1.0, "blue")
        ]

        # Create the choropleth map with detailed hover information
        fig = px.choropleth(
            pivot_df,
            geojson=us_states_geojson,
            locations='state',
            featureidkey='properties.name',
            color='comparative_score',
            color_continuous_scale=custom_color_scale,
            color_continuous_midpoint=0,
            range_color=[-abs_max, abs_max],
            scope="usa",
            labels={'comparative_score': 'Comparative Score (Biden - Trump)'},
            hover_data={
                'state': True,
                'comparative_score': ':.4f',
                'total_tweets': True,
                'average_engagement': ':.2f',
                'normalized_score_biden': ':.4f',
                'normalized_score_trump': ':.4f'
            },
            template="plotly_dark"
            
        )

        fig.update_geos(fitbounds="locations", visible=False)
        fig.update_layout(
            autosize=True,
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            coloraxis_colorbar={
                'title': 'Comparative Score',
                'ticksuffix': '',
                'showticksuffix': 'last'
            }
        )
        
        st.write("**Choropleth Map of Comparative Sentiment Scores:**")
        st.plotly_chart(fig, use_container_width=True)

    def main(self) -> None:
        """
        Main function that initializes filters, applies them, and generates the choropleth map.
        """
        # Call sidebar_filters and apply_filters using StreamlitFilters
        self.filters.sidebar_filters()
        self.filters.apply_filters()

        # Only generate the choropleth map if filtered data is available
        try: 
            self.create_choropleth_map(
                df=self.filters.filtered_data,
                geojson_url=self.geojson_url,
                geojson_states=self.geojson_state_names,
                date_range=self.filters.selected_date_range
            )
        except Exception as e:
            st.error(f"Error generating choropleth map: {e}")

if __name__ == "__main__":
    ChoroplethMapPage()
