import pytest
import pandas as pd
import re
from langdetect import detect, LangDetectException
import emoji
from src.preprocessing import tweet_normalize

# Sample DataFrame to be used in the tests
@pytest.fixture
def sample_dataframe():
    data = {
        'created_date': ['2020-10-15', '2020-10-15'],  # separate out created_at
        'created_time': ['00:00:02', '00:00:08'], # separate out created_at
        'tweet_id': [132.0, 133.0],
        'tweet': ['#Trump: As a student...', 'You get a tie!'],
        'likes': ['2', '4'],
        'retweet_count': [10.0, 20.0],
        'source': ['Twitter Web App', 'Twitter for iPhone'],
        'user_id': ['8436472', '47413798'],
        'user_name': ['snarke', 'Rana Abtar'],
        'user_screen_name': ['snarke', 'Ranaabtar'],
        'user_description': ['Freelance writer...', 'Washington Correspondent...'],
        'days_from_join_date': [7, 30], # user_join_date - created_date 
        'user_join_date': ['2007-08-26 05:56:11', '2009-06-15 19:05:35'],
        'user_followers_count': [200, 300],
        'user_location': ['Portland', 'Washington DC'],
        'lat': [-122.6741949, -77.0365581],
        'long': ['-122.6741949', '-77.0365581'],
        'city': ['Portland', 'Washington'],
        'country': ['United States of America', 'United States of America'], 
        'continent': ['North America', 'North America'],
        'state': ['Oregon', 'District of Columbia'],
        'state_code': ['OR', 'DC'],
        'days_from_collection' : [7, 10], # collected_at - created_date
        'hashtag': ['trump', 'biden'],
        'clean_tweet': ['#Trump: As a student...', 'You get a tie!'],
        'no_stopwords': [['#trump', 'student', 'used'], ['#trump', 'rally', '#iowa']]
    }
    return pd.DataFrame(data)

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
        'created_time': 'object',
        'tweet_id': 'float',
        'tweet': 'object',
        'likes': 'object',
        'retweet_count': 'float',
        'source': 'object',
        'user_id': 'object',
        'user_name': 'object',
        'user_screen_name': 'object',
        'user_description': 'object',
        'days_from_join_date': 'int64',
        'user_join_date': 'object',
        'user_followers_count': 'int64',
        'user_location': 'object',
        'lat': 'float',
        'long': 'object',
        'city': 'object',
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

    # Ensure 'country' and 'continent' columns have only expected values
    if 'country' in sample_dataframe.columns:
        assert (sample_dataframe['country'] == 'United States of America').all(), "Column 'country' contains unexpected values"
    if 'continent' in sample_dataframe.columns:
        assert (sample_dataframe['continent'] == 'North America').all(), "Column 'continent' contains unexpected values"

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
            # Test if tweet contains no emojis
            assert contains_no_emojis(tweet), f"Tweet '{tweet}' contains emojis"
            # Test if tweet normalization works as intended
            normalized_tweet = tweet_normalize(tweet)
            assert '@user' in normalized_tweet or 'http' not in normalized_tweet, f"Tweet '{tweet}' did not normalize correctly"
            assert '  ' not in normalized_tweet, f"Tweet '{tweet}' contains extra whitespace after normalization"