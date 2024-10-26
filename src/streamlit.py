# streamlit.py

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
from datetime import datetime, timedelta, date
from src.llm_analyzer import LLMAnalyzer  # New Import
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
- **Choropleth Map:** Visualize comparative sentiment scores between hashtags by state.
- **Time Series Analysis:** Track sentiment trends over time for selected hashtags.
- **Word Clouds:** Explore common words used in tweets for each hashtag.
- **Sentiment Distribution:** Understand the distribution of sentiment scores by hashtag.
- **User Influence:** Analyze the relationship between user followers and sentiment.
- **Latest Day Analysis:** Perform in-depth analysis on the latest day's data, including anomaly detection, key topics, and key trends.

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
    data['created_date'] = pd.to_datetime(data['created_date'])
    # run metrics calculation
    data['engagement'] = engagement_score(data['likes'], data['retweet_count'], data['user_followers_count'])
    data['normalized_score'] = normalization(data['engagement'], data['sentiment'], data['confidence'])
    # cast again for the new columns
    data = cast_data_type(data)
    return data

# Load configuration
config = load_config('conf/config.yaml')

# Load data
data = load_and_cast_data(config)

# Initialize LLMAnalyzer
openai_api_key = config['openai']['api_key']
llm_analyzer = LLMAnalyzer(openai_api_key=openai_api_key)
# check if the model is working
print(llm_analyzer.test_model())

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
    
# Add Time Filter
with st.sidebar.expander("🕒 Time Filter"):
    min_date = data['created_date'].min().date()
    max_date = data['created_date'].max().date()
    selected_date_range = st.date_input(
        "Select Date Range:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Ensure that the selected_date_range is a tuple of two dates
    if isinstance(selected_date_range, date):
        selected_date_range = (selected_date_range, selected_date_range)
    elif len(selected_date_range) != 2:
        st.error("Please select a start and end date.")

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
data['created_date'] = data['created_date'].dt.date

# Apply filters based on sidebar selections
filtered_data = data[
    (data['hashtag'].isin(selected_hashtag)) &
    (data['days_from_join_date'] >= selected_days) &
    (data['user_followers_count'] >= min_followers) &
    (data['created_date'] >= selected_date_range[0]) &
    (data['created_date'] <= selected_date_range[1])
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
# 7. Latest Day Analysis (Updated Section)
# -------------------------
st.header("📅 Latest Day Analysis")

if not filtered_data.empty:
    # Filter data for the latest day
    latest_day_data = llm_analyzer.filter_latest_day(filtered_data)
    columns_to_analyze = ['clean_tweet', 'engagement']
    
    latest_day_data = latest_day_data[columns_to_analyze]
    latest_date = filtered_data['created_date'].max()
    
    # Limiting tweets for testing to prevent token overflow
    latest_day_data = latest_day_data.head(10)
    st.subheader(f"Analysis for {latest_date}")

    if latest_day_data.empty:
        st.warning("No data available for the latest day.")
    else:
        with st.spinner("Performing analysis..."):
            # Generate analysis report using LLMAnalyzer
            analysis_report = llm_analyzer.generate_daily_report(latest_day_data)
        
        if analysis_report:
            # Display the structured report
            st.markdown(f"### 📄 Report for {latest_date}")
            st.json(analysis_report.dict())  # Displaying report as JSON for readability
else:
    st.warning("No data available after applying the selected filters.")

# -------------------------
# 8. Comparative Choropleth Map
# -------------------------
def create_comparative_choropleth_map(df: pd.DataFrame, geojson_url: str, geojson_states: list, date_range: tuple) -> None:
    """
    Create and display a choropleth map showing the comparative sentiment scores between 'biden' and 'trump' by state.

    Args:
        df (pd.DataFrame): The filtered data.
        geojson_url (str): URL to the GeoJSON file.
        geojson_states (list): List of state names from the GeoJSON.
        date_range (tuple): Selected date range.
    """
    st.subheader(f"🌎 Comparative Sentiment Scores by State\n🕒 From {date_range[0]} to {date_range[1]}")

    # Clean state names
    df['state'] = df['state'].str.strip().str.title()

    # Calculate average normalized scores per state and hashtag
    state_hashtag_scores = df.groupby(['state', 'hashtag'])['normalized_score'].mean().reset_index()

    # Pivot the DataFrame to have 'trump' and 'biden' as columns
    pivot_df = state_hashtag_scores.pivot(index='state', columns='hashtag', values='normalized_score').reset_index()

    # Ensure both 'trump' and 'biden' columns are present
    if 'trump' not in pivot_df.columns:
        pivot_df['trump'] = 0.0
    if 'biden' not in pivot_df.columns:
        pivot_df['biden'] = 0.0

    # Calculate comparative score
    pivot_df['comparative_score'] = pivot_df['biden'] - pivot_df['trump']

    # Exclude states not present in GeoJSON
    pivot_df = pivot_df[pivot_df['state'].isin(geojson_states)]

    # Display the pivot_df DataFrame for debugging
    st.write("**Comparative Scores DataFrame:**")
    st.dataframe(pivot_df[['state', 'biden', 'trump', 'comparative_score']])

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

    # Create the choropleth map with centered color scale
    fig = px.choropleth(
        pivot_df,
        geojson=us_states_geojson,
        locations='state',
        featureidkey='properties.name',
        color='comparative_score',
        color_continuous_scale=custom_color_scale,
        color_continuous_midpoint=0,  # Center the color scale at zero
        range_color=[-abs_max, abs_max],  # Symmetric range
        scope="usa",
        labels={'comparative_score': 'Comparative Score (Biden - Trump)'},
        hover_data={'state': True, 'comparative_score': ':.4f'}
    )

    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        coloraxis_colorbar={
            'title': 'Comparative Score',
            'ticksuffix': '',
            'showticksuffix': 'last'
        }
    )

    st.plotly_chart(fig, use_container_width=True)

# Create Comparative Choropleth Map
create_comparative_choropleth_map(filtered_data, geojson_url, geojson_state_names, selected_date_range)

# -------------------------
# 9. Sentiment Distribution by Hashtag
# -------------------------
def sentiment_distribution_by_hashtag(df: pd.DataFrame) -> None:
    """
    Display a histogram of sentiment scores for each hashtag.

    Args:
        df (pd.DataFrame): The filtered data.
    """
    st.subheader("📊 Sentiment Score Distribution by Hashtag")

    if df.empty:
        st.warning("No data available to display the sentiment distribution.")
        return

    hashtags = df['hashtag'].unique()
    for hashtag in hashtags:
        st.markdown(f"#### #{hashtag.capitalize()}")
        hashtag_data = df[df['hashtag'] == hashtag]
        fig = px.histogram(
            hashtag_data,
            x='normalized_score',
            nbins=50,
            title=f"Distribution of Normalized Sentiment Scores for #{hashtag}",
            labels={'normalized_score': 'Normalized Sentiment Score'},
            color_discrete_sequence=['#636EFA']
        )
        st.plotly_chart(fig, use_container_width=True)

# Display Sentiment Distribution by Hashtag
sentiment_distribution_by_hashtag(filtered_data)

# -------------------------
# 10. Time Series Analysis
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
    selected_state = st.selectbox("Select State for Time Series Analysis:", options=states)

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
# 11. Word Cloud Generation
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
# 12. User Influence Analysis
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
# 13. Footer
# -------------------------
st.markdown("""
---
**Data Source:** [US Election 2020 Tweets](https://www.kaggle.com/datasets/manchunhui/us-election-2020-tweets)

<small>**Developed by:** Team29 - Anthony, Chiyang, Kwong Jia Ying</small>
---
""", unsafe_allow_html=True)
