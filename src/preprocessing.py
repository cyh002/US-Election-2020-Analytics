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
import yaml
import datetime as dt

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
    data = pd.read_csv(path, header=0, lineterminator='\n', low_memory=False, encoding='utf-8-sig')
    return data

def cast_data_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Casts the columns of the DataFrame to appropriate data types based on their content.

    Parameters:
    df (pd.DataFrame): The input DataFrame with mixed data types.

    Returns:
    pd.DataFrame: DataFrame with corrected data types.
    """
    # Define the column casting
    df['tweet_id'] = df['tweet_id'].astype(str)  # Cast tweet_id to string to preserve precision
    df['likes'] = df['likes'].astype(np.int64)  # Cast likes to integer
    df['retweet_count'] = df['retweet_count'].astype(np.int64)  # Cast retweet_count to integer
    df['user_id'] = df['user_id'].astype(str)  # Cast user_id to string to preserve precision
    df['user_id_post_count'] = df['user_id_post_count'].astype(np.int64)  # Cast user_id_post_count to integer
    df['days_from_join_date'] = df['days_from_join_date'].astype(np.int64)  # Cast days_from_join_date to integer
    df['user_followers_count'] = df['user_followers_count'].astype(np.int64)  # Cast user_followers_count to integer
    df['sentiment'] = df['sentiment'].astype(np.int64)  # Cast sentiment to integer
    df['confidence'] = df['confidence'].astype(float)  # Cast confidence to float
    if 'engagement' in df.columns:
        print(f"Engagement column found in the DataFrame. Casting 'engagement' to float.")
        df['engagement'] = df['engagement'].astype(float)  # Cast engagement to float
    if 'normalized_score' in df.columns:
        print(f"Normalized_score column found in the DataFrame. Casting 'normalized_score' to float.")
        df['normalized_score'] = df['normalized_score'].astype(float)  # Cast normalized_scores to float
    
    # Ensure string types for text fields
    df['source'] = df['source'].astype(str)
    df['user_description'] = df['user_description'].astype(str)
    df['state'] = df['state'].astype(str)
    df['hashtag'] = df['hashtag'].astype(str)
    df['clean_tweet'] = df['clean_tweet'].astype(str)
    df['no_stopwords'] = df['no_stopwords'].astype(str)
    df['created_date'] = pd.to_datetime(df['created_date'], errors='coerce').apply(lambda x: x.date())  # Cast created_date to date
    df['created_time'] = pd.to_datetime(df['created_time'], format='%H:%M:%S', errors='coerce').dt.time  # Cast created_time to time
 
    return df
    

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
    if 'hashtag' in data.columns:
        data.drop('hashtag', axis=1, inplace=True)
    data['hashtag'] = name
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

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip whitespace and normalize column names to avoid mismatches.
    """
    df.columns = df.columns.str.strip().str.lower().astype(str)
    return df

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
    data = data[data['country'].isin(country) & data['state'].notna()].copy()
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
    data = data[data['state'].isin(state_names)].copy()
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
    DetectorFactory.seed = 0
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

def remove_duplicates_retweets(data: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """
    Remove duplicate tweets and retwets.

    Args:
        tweet (str): Tweet text to process.

    Returns:
        list: List of tweets that are non_duplicates. 
    """
    assert col_name in data.columns, f"Error: {col_name} is not a valid column. Available columns are: {list(data.columns)}"
    assert isinstance(col_name, str), "Error: col_name must be a string"

    #Remove all the retweets
    data = data[~data[col_name].str.startswith('RT')]

    #Drop the first occurrence of each duplicate
    data = data.drop_duplicates(subset = col_name, keep = 'first')
    
    return data

def remove_duplicates_renamed(data:pd.DataFrame, col_name: str)-> pd.DataFrame:
    """
    Remove duplicate tweets and retwets.

    Args:
        tweet (str): Tweet text to process.

    Returns:
        list: List of tweets that are non_duplicates. 
    """
    assert col_name in data.columns, f"Error: {col_name} is not a valid column. Available columns are: {list(data.columns)}"
    assert isinstance(col_name, str), "Error: col_name must be a string"

    #Identify all the duplicates
    duplicates = data[data.duplicated(subset = col_name,keep=False)]

    #Rename all the duplicates as 'both'
    data.loc[duplicates.index,'hashtag'] ='both'

    #Drop the first occurrence of each duplicate
    data = data.drop_duplicates(subset = col_name, keep = 'last')
    
    return data

def created_date_and_time(df:pd.DataFrame, col_name:str) -> pd.DataFrame:
    """
    Extract date and time from a datetime column in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        col_name (str): The name of the column containing datetime values.

    Returns:
        pd.DataFrame: A DataFrame with two new columns, 'created_date' and 'created_time', 
                    representing the extracted date and time respectively.
    """
    #Change the column to datetime
    df[col_name] = pd.to_datetime(df[col_name], format='%Y-%m-%d %H:%M:%S')

    #Get the corresponding date and time
    df['created_date'] = pd.to_datetime(df[col_name].dt.date)
    df['created_time'] = df[col_name].dt.strftime('%H:%M:%S')
    return df

def days_from_joined_date(df:pd.DataFrame, user_join_column:str, created_column:str)-> pd.DataFrame:
    """
    Calculate the number of days from the user's join date to the tweet creation date.

    Args:
        df (pd.DataFrame): The input DataFrame.
        user_join_column (str): The name of the column containing user join dates.
        created_column (str): The name of the column containing tweet creation dates.

    Returns:
        pd.DataFrame: A DataFrame with a new column 'days_from_join_date' 
                    indicating the number of days between the two dates.

    Raises:
        AssertionError: If any 'created_column' value is earlier than 'user_join_column'.
    """
    #Change the column to date
    df[user_join_column] = pd.to_datetime(df[user_join_column]).dt.date
    df[created_column] = pd.to_datetime(df[created_column]).dt.date


    #Write an assert statement to check that all joined_date is older that created_date
    assert (df[user_join_column] <= df[created_column]).all(), \
    f"Found rows where '{user_join_column}' is later than '{created_column}'"

    #Find the number of days from joined date
    df['days_from_join_date'] = (pd.to_datetime(df[created_column]) - pd.to_datetime(df[user_join_column])).dt.days
    return df

def userid_post_count(df:pd.DataFrame, col_name:str) -> pd.DataFrame:
    """
    Calculate the number of posts by each user and add it as a new column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        col_name (str): The name of the column containing user IDs.

    Returns:
        pd.DataFrame: A DataFrame with a new column 'user_id_post_count' 
                    showing the count of posts for each user.
    """

    #Get the value count for each user
    post_count = df[col_name].value_counts()

    #Map the post counts back to the original DataFrame as a new column
    df['user_id_post_count'] = df[col_name].map(post_count)
    return df

def columns_to_keep(df:pd.DataFrame, column_datatype_list: list) -> pd.DataFrame:
    """
    Ensure the DataFrame contains necessary columns with specified data types 
    and keeps only the required columns.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_datatype_list (list): A list of dictionaries, each containing:
            - 'name' (str): The column name.
            - 'dtype' (str): The data type (e.g., 'int', 'float', 'datetime', 'string', 'list').

    Returns:
        pd.DataFrame: A DataFrame containing only the required columns with 
                    values converted to the specified data types. 
                    Missing columns are filled with None.

    Raises:
        ValueError: If a data type in the configuration is not supported.
    """
    #Get the required columns
    required_columns = [col['name'] for col in column_datatype_list]
    # Create a dictionary of {column_name: dtype} from the config
    dtype_mapping = {col['name']: col['dtype'] for col in column_datatype_list}

    # Ensure the DataFrame has all necessary columns, or add them with default values
    for col_name in required_columns:
        if col_name not in df.columns:
            df[col_name] = None  # Add missing columns with None

    for col_name, dtype in dtype_mapping.items():
        if dtype == 'datetime':
            df[col_name] = pd.to_datetime(df[col_name], errors='coerce')
        elif dtype == 'int':
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce').astype(np.int64)
        elif dtype == 'float':
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce').astype(float)
        elif dtype == 'string':
            df[col_name] = df[col_name].astype(str)
        else:
            # If dtype is 'list', skip processing and leave the column as-is
            continue
    
    # Keep only the required columns
    df = df[required_columns]

    return df

def run_preprocessing_pipeline(config_path: str, trump_path: str = None, biden_path: str = None, output_path: str = None, output_path_no_dups: str = None) -> (pd.DataFrame, pd.DataFrame):
    """
    Main function to run the preprocessing pipeline.

    This function loads the configuration, processes the tweet data, and outputs a preprocessed dataset.

    Args:
        config_path (str): Path to the configuration file.
        trump_path (str): Path to the Trump data CSV file.
        biden_path (str): Path to the Biden data CSV file.
        output_path (str): Path to save the preprocessed data CSV file.
        output_path_no_dup(str): path to save the preprocessed data without duplicates in CSV file.
        
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

    print('Removing duplicates and retweets from trump and biden dataset...')    
    #Drop the retweets and duplicates from each dataset
    trump_data = remove_duplicates_retweets(trump_data,'tweet')
    biden_data = remove_duplicates_retweets(biden_data,'tweet')

    trump_data = clean_columns(trump_data)
    biden_data = clean_columns(biden_data)

    # Merge datasets
    print("Merging datasets...")
    combined_data = merge_df(trump_data, biden_data)
    combined_data.reset_index(drop=True, inplace=True)
  
    #Remove the duplicates
    print('Removing duplicates and renamed for merged dataset...')
    #Keep all the first tweets
    combined_data = remove_duplicates_renamed(combined_data, 'tweet')

    # Filter by US country and states
    print("Filtering data by country and state...")
    us_tweets = filter_country(combined_data, config['filter']['country'])
    us_state = filter_state(us_tweets, config['filter']['state_names'])
    
    # Preprocess tweets
    print("Preprocessing tweets...")
    us_state.loc[:, 'clean_tweet'] = us_state['tweet'].apply(preprocess_tweet)
    print(us_state.columns)
    
    # Detect English tweets
    print("Detecting english tweets...")
    english_tweets = us_state.loc[us_state['clean_tweet'].apply(detect_english_tweets)]

    # # Tokenize and remove stopwords
    print("Tokenizing and removing stopwords...")
    english_tweets.loc[:, 'no_stopwords'] = english_tweets['clean_tweet'].apply(remove_stopwords)

    #Drop duplicated clean_tweet
    print("Removing duplicate clean_tweet")
    english_tweets = english_tweets.drop_duplicates(subset="clean_tweet", keep = "first")

    #Feature Engineering
    print("Splitting the 'created_at' to 'created_date' and 'created_time'...")
    final_results = created_date_and_time(english_tweets, 'created_at')

    print("Finding days between tweet created date and user_joined_date...")
    final_results = days_from_joined_date(final_results, 'user_join_date', 'created_date')

    print("Finding the number of post per user")
    final_results = userid_post_count(final_results, "user_id")

    #Cast the dataframe back to the right datatype and get the columns needed
    print("Casting the dataframe back to the right datatypes...")
    final_results = columns_to_keep(final_results, config['columns_to_keep'])

    # Save datasets (Optional)
    print("Saving preprocessed data with duplicates...")
    final_results.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Preprocessed data with duplicates saved to {output_path}")
    
    print("Preprocessing pipeline completed.")
    
    return english_tweets

if __name__ == '__main__':
    config_path = 'conf/config.yaml'
    final_results = run_preprocessing_pipeline(config_path)