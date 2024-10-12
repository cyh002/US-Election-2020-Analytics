# to run : 
# python -m src.preprocessing
import pandas as pd
import numpy as np
import re
from collections import Counter
import emoji
from langdetect import detect, DetectorFactory, LangDetectException
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import yaml

# Load config file
def load_config(path: str = 'conf/config.yaml') -> dict:
    """
    Load configuration parameters from a YAML file.

    Args:
        path (str): Path to the YAML configuration file. Defaults to 'config.yaml'.

    Returns:
        dict: Configuration parameters as a dictionary.
    """
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Load data function
def load_data(path: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame with the data from the CSV file.
    """
    data = pd.read_csv(path, header=0, lineterminator='\n')
    return data

# Add candidate name to data
def candidate_name(data: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Add a column for the candidate's name to the DataFrame.

    Args:
        data (pd.DataFrame): DataFrame containing the tweet data.
        name (str): Candidate name to add.

    Returns:
        pd.DataFrame: DataFrame with an additional 'candidate_name' column.
    """
    assert isinstance(name, str), 'Strings only!'
    if 'candidate_name' in data.columns:
        data.drop('candidate_name', axis=1, inplace=True)
    data['candidate_name'] = name
    return data

# Merge two dataframes
def merge_df(data1: pd.DataFrame, data2: pd.DataFrame) -> pd.DataFrame:
    """
    Merge two DataFrames, ensuring they have the same column headers.

    Args:
        data1 (pd.DataFrame): First DataFrame.
        data2 (pd.DataFrame): Second DataFrame.

    Returns:
        pd.DataFrame: Merged DataFrame.

    Raises:
        AssertionError: If the column headers do not match.
    """
    assert list(data1.columns) == list(data2.columns), "The dataframes do not have the same column headers."
    combine_data = pd.concat([data1, data2], ignore_index=True)
    return combine_data

# Filter by country and state
def filter_country(data: pd.DataFrame, country: list) -> pd.DataFrame:
    """
    Filter the DataFrame for specific countries and ensure the 'state' column is not empty.

    Args:
        data (pd.DataFrame): DataFrame containing the tweet data.
        country (list): List of country names to filter by.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    assert isinstance(country, list) and all(isinstance(c, str) for c in country), "Provide a list of strings as the country filter."
    data = data[data['country'].isin(country) & data['state'].notna()]
    return data

def filter_state(data: pd.DataFrame, state_names: list) -> pd.DataFrame:
    """
    Filter the DataFrame to only include specific U.S. states.

    Args:
        data (pd.DataFrame): DataFrame containing the tweet data.
        state_names (list): List of U.S. state names to filter by.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    assert isinstance(state_names, list) and all(isinstance(c, str) for c in state_names), "Provide a list of strings as the state filter."
    data = data[data['state'].isin(state_names)]
    return data

# Tweet cleaning and preprocessing functions
def replace_emoji(tweet: str) -> str:
    """
    Replace all emojis in a tweet with their corresponding text.

    Args:
        tweet (str): Original tweet containing emojis.

    Returns:
        str: Tweet with emojis replaced by text.
    """
    return emoji.demojize(tweet, delimiters=(" ", " "))

def tweet_normalize(tweet: str) -> str:
    """
    Normalize a tweet by replacing user mentions, removing URLs, and cleaning whitespace.

    Args:
        tweet (str): Original tweet text.

    Returns:
        str: Normalized tweet text.
    """
    tweet = re.sub(r'@\w+', '@user', tweet)
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    tweet = tweet.strip()
    tweet = re.sub(r'\s+', ' ', tweet)
    return tweet

def preprocess_tweet(tweet: str) -> str:
    """
    Preprocess a tweet by removing emojis and normalizing the text.

    Args:
        tweet (str): Original tweet text.

    Returns:
        str: Preprocessed tweet.
    """
    tweet = replace_emoji(tweet)
    tweet = tweet_normalize(tweet)
    return tweet

# Detect English tweets
def detect_english_tweets(text: str) -> bool:
    """
    Detect if a tweet is in English.

    Args:
        text (str): Tweet text.

    Returns:
        bool: True if the tweet is in English, False otherwise.
    """
    try:
        lang = detect(text)
        return lang == 'en'
    except LangDetectException:
        return False

# Remove stopwords for tokenizing
def remove_stopwords(tweet: str) -> list:
    """
    Tokenize and remove stopwords from a tweet.

    Args:
        tweet (str): Tweet text to process.

    Returns:
        list: List of cleaned tokens (words) from the tweet.
    """
    tweet = tweet.lower()
    tokenizer = TweetTokenizer()
    tokenized = tokenizer.tokenize(tweet)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokenized = [lemmatizer.lemmatize(token) 
                 for token in tokenized 
                 if (token.isalpha() or token.startswith('#')) 
                 and len(token) > 3 
                 and token not in stop_words 
                 and not token.startswith('@')]
    return tokenized

# Create Document Term Matrix (DTM)
def create_dtm(data: pd.DataFrame, text_column: str, max_features=1000, max_df=0.5) -> (pd.DataFrame, CountVectorizer):
    """
    Create a Document-Term Matrix (DTM) using the given text column from a DataFrame.

    Args:
        data (pd.DataFrame): DataFrame containing the text data.
        text_column (str): Column name containing the preprocessed text.
        max_features (int): Maximum number of features (terms) to include in the DTM.
        max_df (float): Maximum document frequency to consider a term relevant.

    Returns:
        pd.DataFrame: Document-Term Matrix as a DataFrame.
        CountVectorizer: Fitted CountVectorizer object.
    """
    vectorizer = CountVectorizer(token_pattern=r'\b[a-zA-Z_]{3,}[a-zA-Z]*\b', max_features=max_features, max_df=max_df)
    dtm_bow = vectorizer.fit_transform(data[text_column])
    dtm_bow_df = pd.DataFrame(dtm_bow.toarray(), columns=vectorizer.get_feature_names_out())
    return dtm_bow_df, vectorizer

def run_preprocessing_pipeline(config_path: str, trump_path: str = None, biden_path: str = None, output_path: str = None) -> (pd.DataFrame, CountVectorizer, pd.DataFrame):
    """
    Main function to run the preprocessing pipeline.

    This function loads the configuration, processes the tweet data, and outputs a preprocessed dataset.

    Args:
        config_path (str): Path to the configuration file.
        trump_path (str): Path to the Trump data CSV file.
        biden_path (str): Path to the Biden data CSV file.
        output_path (str): Path to save the preprocessed data CSV file.
        
    Returns:
        pd.DataFrame: Document-Term Matrix (DTM) of the preprocessed tweet data.
        CountVectorizer: Fitted CountVectorizer object.
        pd.DataFrame: Preprocessed tweet data.
    """
    # Load configuration
    config = load_config(config_path)
    
    if trump_path is None:
        trump_path = config['data']['trump_path']
    
    if biden_path is None:
        biden_path = config['data']['biden_path']
    
    if output_path is None:
        output_path = config['data']['output_path']

    # Load Trump and Biden data
    print("Loading data...")
    trump_data = load_data(trump_path)
    biden_data = load_data(biden_path)

    # Add candidate name
    print("Adding candidate names...")
    trump_data = candidate_name(trump_data, 'trump')
    biden_data = candidate_name(biden_data, 'biden')

    # Merge datasets
    print("Merging datasets...")
    combined_data = merge_df(trump_data, biden_data)

    # Filter by US country and states
    print("Filtering data by country and state...")
    us_tweets = filter_country(combined_data, config['filter']['country'])
    us_state = filter_state(us_tweets, config['filter']['state_names'])
    
    # Preprocess tweets
    print("Preprocessing tweets...")
    us_state.loc[:, 'clean_tweet'] = us_state['tweet'].apply(preprocess_tweet)
    
    # Detect English tweets
    print("Detecting English tweets...")
    english_tweets = us_state.loc[us_state['clean_tweet'].apply(detect_english_tweets)]

    # Tokenize and remove stopwords
    print("Tokenizing and removing stopwords...")
    english_tweets.loc[:, 'no_stopwords'] = english_tweets['clean_tweet'].apply(remove_stopwords)

    # Create Document Term Matrix
    print("Creating Document-Term Matrix (DTM)...")
    dtm, vectorizer = create_dtm(english_tweets, 'no_stopwords')

    # Save preprocessed data (Optional)
    print("Saving preprocessed data...")
    english_tweets.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")
    
    print("Preprocessing pipeline completed.")
    return dtm, vectorizer, english_tweets

if __name__ == '__main__':
    config_path = 'conf/config.yaml'
    dtm, vectorizer, english_tweets = run_preprocessing_pipeline(config_path)