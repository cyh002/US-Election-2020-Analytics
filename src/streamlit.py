import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict
import json
import requests
from src.preprocessing import load_config, load_data, cast_data_type
from src.streamlit_app.misc_app_utils import get_geojson_state_names, compare_state_names
import os
from src.misc_utils import engagement_score, normalization, normalize_scores

# ------------------------------------------------------------
# Streamlit Twitter Sentiment Analysis Dashboard
# ------------------------------------------------------------

# -------------------------
# 1. Page Configuration
# -------------------------
st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon=":bar_chart:",
    layout="wide"
)

# -------------------------
# 2. Introduction
# -------------------------
st.markdown("""
# Twitter Sentiment Analysis Dashboard

Welcome to the Twitter Sentiment Analysis Dashboard! This application visualizes the sentiment of tweets across different US states and hashtags. Explore the data through interactive maps, time series plots, word clouds, and more.

**Features:**
- **Choropleth Map:** Visualize average normalized sentiment scores by state.
- **Time Series Analysis:** Track sentiment trends over time for selected hashtags.
- **Word Clouds:** Explore common words used in tweets for each hashtag.
- **Sentiment Distribution:** Understand the distribution of sentiment scores.
- **User Influence:** Analyze the relationship between user followers and sentiment.

Use the sidebar filters to customize your view and dive deep into the data.
""")

# -------------------------
# 3. Load Configuration and Data
# -------------------------
@st.cache_data
def load_and_cast_data(config: Dict) -> pd.DataFrame:
    """
    Load and cast data types of the CSV data based on the configuration.

    Args:
        config (Dict): Configuration dictionary.

    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    data = pd.read_csv(config['streamlit']['data'])
    # cast features to appropriate data types first
    data = cast_data_type(data) 
    # run metrics calculation
    data['engagement'] = engagement_score(data['likes'], data['retweet_count'], data['user_followers_count'])
    data['normalized_score'] = normalization(data['engagement'], data['sentiment'], data['confidence'])
    data['normalized_score'] = normalize_scores(data['normalized_score'])
    # cast again for the new columns
    data = cast_data_type(data)
    return data

# Load configuration
config = load_config('conf/config.yaml')

# Load data
data = load_and_cast_data(config)

# -------------------------
# 4. Sidebar Filters
# -------------------------
st.sidebar.header("Filters")

# Organize filters within expandable sections for better UI
with st.sidebar.expander("📊 Filter Options"):
    # Hashtag Filter
    hashtags = data['hashtag'].unique().tolist()
    selected_hashtag = st.multiselect(
        "Select Hashtag(s):",
        options=hashtags,
        default=hashtags
    )
    
    # Days from Join Date Slider
    min_days = int(data['days_from_join_date'].min())
    max_days = int(data['days_from_join_date'].max())
    selected_days = st.slider(
        "Select minimum days since user joined:",
        min_value=min_days,
        max_value=max_days,
        value=min_days
    )
    
    # User Followers Count Filter
    min_followers = st.number_input(
        "Minimum user followers count:",
        min_value=0,
        value=10,
        step=1
    )

# Fetch GeoJSON state names
geojson_url = config['streamlit']['geojson_url']
geojson_state_names = get_geojson_state_names(geojson_url)

# Display GeoJSON State Names in Sidebar
with st.sidebar.expander("📍 GeoJSON State Names"):
    st.write(geojson_state_names)

# -------------------------
# 5. Data Filtering
# -------------------------
# Apply filters based on sidebar selections
filtered_data = data[
    (data['hashtag'].isin(selected_hashtag)) &
    (data['days_from_join_date'] >= selected_days) &
    (data['user_followers_count'] >= min_followers)
].dropna(subset=['normalized_score'])

# Compare state names between data and GeoJSON
compare_state_names(filtered_data['state'].unique().tolist(), geojson_state_names)

# Select columns to display to avoid serialization issues
columns_to_display = [
    'state',
    'normalized_score',
    'days_from_join_date',
    'user_followers_count',
    'engagement',
    'created_date',
    'hashtag'
]

# -------------------------
# 6. Display Filtered Data
# -------------------------
st.subheader("🔍 Filtered Data")
st.write(filtered_data[columns_to_display])
st.write(f"**Total Tweets:** {filtered_data.shape[0]}")

# -------------------------
# 7. Choropleth Map
# -------------------------
def create_choropleth_map(df: pd.DataFrame, geojson_url: str, geojson_states: list) -> None:
    """
    Create and display a choropleth map of US states showing average normalized scores.

    Args:
        df (pd.DataFrame): The filtered data.
        geojson_url (str): URL to the GeoJSON file.
        geojson_states (list): List of state names from the GeoJSON.
    """
    st.subheader("🌎 Average Normalized Sentiment Scores by State")

    # Clean state names
    df['state'] = df['state'].str.strip().str.title()

    # Group by state and calculate mean normalized scores
    state_scores = df.groupby('state')['normalized_score'].mean().reset_index()

    # Exclude states not present in GeoJSON
    state_scores = state_scores[state_scores['state'].isin(geojson_states)]

    # Exclude states with zero normalized_score to avoid misleading color mapping
    state_scores = state_scores[state_scores['normalized_score'] != 0.0]

    # Display the state_scores DataFrame for debugging
    st.write("**State Scores DataFrame:**")
    st.dataframe(state_scores)

    if state_scores.empty:
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

    # Determine min and max of 'normalized_score' for symmetric color scaling
    min_score = state_scores['normalized_score'].min()
    max_score = state_scores['normalized_score'].max()
    abs_max = max(abs(min_score), abs(max_score))

    # Define custom color scale: negative red, zero white, positive green
    custom_color_scale = [
        (0.0, "red"),
        (0.5, "white"),
        (1.0, "green")
    ]

    # Create the choropleth map with centered color scale
    fig = px.choropleth(
        state_scores,
        geojson=us_states_geojson,
        locations='state',
        featureidkey='properties.name',
        color='normalized_score',
        color_continuous_scale=custom_color_scale,
        color_continuous_midpoint=0,  # Center the color scale at zero
        range_color=[-abs_max, abs_max],  # Symmetric range
        scope="usa",
        labels={'normalized_score': 'Avg Normalized Score'},
        hover_data={'state': True, 'normalized_score': ':.2f'}
    )

    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    st.plotly_chart(fig, use_container_width=True)

# Create Choropleth Map
create_choropleth_map(filtered_data, geojson_url, geojson_state_names)

# -------------------------
# 8. Time Series Analysis
# -------------------------
def time_series_analysis(df: pd.DataFrame) -> None:
    """
    Display a time series analysis of normalized scores for each hashtag.

    Args:
        df (pd.DataFrame): The filtered data.
    """
    st.subheader("📈 Time Series Analysis")

    # Select State
    states = df['state'].unique().tolist()
    selected_state = st.selectbox("Select State:", options=states)

    # Prepare data for all hashtags
    state_data = df[df['state'] == selected_state]

    if state_data.empty:
        st.warning("No data available for the selected state.")
        return

    # Group by date and hashtag to calculate mean normalized scores
    time_series = state_data.groupby(['created_date', 'hashtag'])['normalized_score'].mean().reset_index()

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
        title=f"Average Normalized Scores Over Time in {selected_state}",
        labels={'created_date': 'Date', 'normalized_score': 'Avg Normalized Score', 'hashtag': 'Hashtag'}
    )

    fig.update_layout(
        legend_title_text='Hashtag',
        margin={"r": 0, "t": 30, "l": 0, "b": 0}
    )

    st.plotly_chart(fig, use_container_width=True)

# Execute Time Series Analysis
time_series_analysis(filtered_data)

# -------------------------
# 9. Word Cloud Generation
# -------------------------
def generate_word_cloud(df: pd.DataFrame, selected_hashtags: List[str]) -> None:
    """
    Generate and display word clouds for each selected hashtag.

    Args:
        df (pd.DataFrame): The filtered data.
        selected_hashtags (List[str]): List of selected hashtags.
    """
    st.subheader("☁️ Word Cloud by Hashtag")
    
    for hashtag in selected_hashtag:
        st.markdown(f"#### #{hashtag.capitalize()}")
        hashtag_data = df[df['hashtag'] == hashtag]
        text = " ".join(tweet for tweet in hashtag_data['clean_tweet'].astype(str))
        
        if text.strip():
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            plt.figure(figsize=(15, 7.5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)
            plt.close()  # Close the figure after rendering
        else:
            st.info(f"No tweets available for #{hashtag}.")

# Generate Word Clouds
generate_word_cloud(filtered_data, selected_hashtag)

# -------------------------
# 10. Sentiment Distribution
# -------------------------
def sentiment_distribution(df: pd.DataFrame) -> None:
    """
    Display a histogram of sentiment scores.

    Args:
        df (pd.DataFrame): The filtered data.
    """
    st.subheader("📊 Sentiment Score Distribution")

    if df.empty:
        st.warning("No data available to display the sentiment distribution.")
        return

    fig = px.histogram(
        df,
        x='normalized_score',
        nbins=50,
        title="Distribution of Normalized Sentiment Scores",
        labels={'normalized_score': 'Normalized Sentiment Score'},
        color_discrete_sequence=['#636EFA']
    )

    st.plotly_chart(fig, use_container_width=True)

# Display Sentiment Distribution
sentiment_distribution(filtered_data)

# -------------------------
# 11. User Influence Analysis
# -------------------------
def user_influence(df: pd.DataFrame) -> None:
    """
    Display a scatter plot of user followers count vs. normalized scores.

    Args:
        df (pd.DataFrame): The filtered data.
    """
    st.subheader("📈 User Influence Analysis")

    if df.empty:
        st.warning("No data available to display the user influence analysis.")
        return

    fig = px.scatter(
        df,
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

# Execute User Influence Analysis
user_influence(filtered_data)

# -------------------------
# 12. Footer
# -------------------------
st.markdown("""
---
**Data Source:** Your Data Source Here  
**Developed by:** Your Name  
**Contact:** [your.email@example.com](mailto:your.email@example.com)
""")
