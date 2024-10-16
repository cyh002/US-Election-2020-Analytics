# to run: 
# python -m pytest -s

import pytest
import pandas as pd
import re
from langdetect import detect, LangDetectException
import emoji
import yaml
from src.preprocessing import tweet_normalize
import os

# Load configuration from config file
with open('conf/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Sample DataFrame to be used in the tests
@pytest.fixture
def sample_dataframe():
    dataset_path = 'data/cleaned/dtm.csv'
    if os.path.exists(dataset_path):
        return pd.read_csv(dataset_path)
    else:
        print("Dataset path: {dataset_path} NOT FOUND.")
    raise FileNotFoundError(f"Dataset path: {dataset_path} NOT FOUND.")

def test_hashtag_values(sample_dataframe):
    # Define allowed values
    allowed_values = {'biden', 'trump', 'both'}
    
    # Check if all values in the 'hashtag' column are in the allowed set
    assert all(sample_dataframe['hashtag'].isin(allowed_values)), (
        f"Invalid values found in 'hashtag' column. Allowed values: {allowed_values}"
    )
def test_data_types(sample_dataframe):
    # Helper function to check column data type
    def check_dtype(df, column, expected_dtype):
        if column in df.columns:
            assert df[column].dtype == expected_dtype, f"Column '{column}' has incorrect dtype: {df[column].dtype}, expected: {expected_dtype}"
        else:
            print(f"Column '{column}' not found in DataFrame. Skipping dtype check.")

    # Expected data types for all columns
    expected_dtypes = {
        'created_date': 'object',
        'created_time': 'object',  # Stored as time object after conversion
        'tweet_id': 'float64',
        'tweet': 'object',
        'likes': 'int64',
        'retweet_count': 'float64',
        'source': 'object',
        'user_id': 'int64',
        'user_name': 'object',
        'user_screen_name': 'object',
        'user_description': 'object',
        'days_from_join_date': 'int64',
        'user_join_date': 'object',  # Could be converted to datetime later if needed
        'user_followers_count': 'int64',
        'user_location': 'object',
        'lat': 'float64',
        'long': 'float64',  # Converted to numeric
        'country': 'object',
        'continent': 'object',
        'state': 'object',
        'state_code': 'object',
        'days_from_collection': 'int64',
        'hashtag': 'object',
        'clean_tweet': 'object',
        'no_stopwords': 'object'
    }

    # Test if the columns have expected data types
    for column, expected_dtype in expected_dtypes.items():
        check_dtype(sample_dataframe, column, expected_dtype)

    # Ensure 'country' and 'state' columns have only expected values from config
    valid_countries = config['filter']['country']
    valid_states = config['filter']['state_names']

    if 'country' in sample_dataframe.columns:
        assert sample_dataframe['country'].isin(valid_countries).all(), "Column 'country' contains unexpected values"
    if 'state' in sample_dataframe.columns:
        assert sample_dataframe['state'].isin(valid_states).all(), "Column 'state' contains unexpected values"

def test_clean_tweet_english_and_normalized(sample_dataframe, sample_count=2):
    # Helper function to check if a tweet contains no emojis
    def contains_no_emojis(tweet):
        # Count the number of emojis in the tweet
        emoji_count = emoji.emoji_count(tweet)
        # Return True if there are no emojis, False otherwise
        return emoji_count == 0

    # Sample some tweets and ensure they are properly processed
    if 'clean_tweet' in sample_dataframe.columns:
        sample_tweets = sample_dataframe['clean_tweet'].sample(sample_count, random_state=42)
        for tweet in sample_tweets:
            try:
                # Test if tweet contains no emojis
                assert contains_no_emojis(tweet), f"Tweet '{tweet}' contains emojis"
                # Test if tweet normalization works as intended
                normalized_tweet = tweet_normalize(tweet)
                assert '@user' in normalized_tweet or 'http' not in normalized_tweet, f"Tweet '{tweet}' did not normalize correctly"
                assert '  ' not in normalized_tweet, f"Tweet '{tweet}' contains extra whitespace after normalization"
            except AssertionError as e:
                print(f"Error for tweet: {tweet}")
                raise e