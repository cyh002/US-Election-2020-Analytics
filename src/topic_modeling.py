#Import all the libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import yaml
from preprocessing import load_data, load_config

def get_vectorizer(vectorizer_type:str, max_features: int, max_df: float) -> object:
    """
    Create either a CountVectorizer or TfidfVectorizer based on the configuration.

    Args:
        config (dict): Configuration dictionary containing vectorizer settings.

    Returns:
        vectorizer: A CountVectorizer or TfidfVectorizer object based on the config.
    """
    if vectorizer_type == 'count':
        vectorizer = CountVectorizer(token_pattern=r'\b[a-zA-Z_]{3,}[a-zA-Z]*\b',
                                     max_features=max_features, max_df=max_df)
    elif vectorizer_type == 'tfidf':
        vectorizer = TfidfVectorizer(token_pattern=r'\b[a-zA-Z_]{3,}[a-zA-Z]*\b',
                                     max_features=max_features, max_df=max_df)
    else:
        raise ValueError(f"Invalid vectorizer type: {vectorizer_type}")
    
    return vectorizer

# Create Document Term Matrix (DTM)
def create_dtm(data: pd.DataFrame, text_column: str, vectorizer) -> (pd.DataFrame, object):
    """
    Create a Document-Term Matrix (DTM) using the given text column from a DataFrame.

    Args:
        data (pd.DataFrame): DataFrame containing the text data.
        text_column (str): Column name containing the preprocessed text.
        max_features (int): Maximum number of features (terms) to include in the DTM.
        max_df (float): Maximum document frequency to consider a term relevant.

    Returns:
        pd.DataFrame: Document-Term Matrix as a DataFrame.
    """
    dtm_bow = vectorizer.fit_transform(data[text_column])
    dtm_bow_df = pd.DataFrame(dtm_bow.toarray(), columns=vectorizer.get_feature_names_out())
    return dtm_bow_df

def topic_preprocess(config_path, combine_nodups_path:str = None, combine_dups_path: str = None, text_column:str = 'no_stopwords')-> dict:
    """
    Preprocess Biden, Trump, and combined datasets for topic modeling using CountVectorizer and TfidfVectorizer.
    """
    # Load configuration
    config = load_config(config_path)
    
    if combine_dups_path is None:
        combine_dups_path = config['data']['output_path']

    if combine_nodups_path is None:
        combine_nodups_path = config['data']['output_path_no_dups']

    #Load all the dataset
    print("Loading dataset...", flush=True)
    combine_nodups = load_data(combine_nodups_path)
    combine_dups = load_data(combine_dups_path)

    #Strip out all the whitespace or special characters
    combine_nodups.columns = combine_nodups.columns.str.strip()
    combine_dups.columns = combine_dups.columns.str.strip()

    #Split the dataset into trump and biden
    trump_data = combine_nodups[combine_nodups['candidate_name'] == 'trump']
    biden_data = combine_nodups[combine_nodups['candidate_name']=='biden']

    #Initialize a new dictionary
    results ={}

    #Get all the vectorizers
    print("Processing the CounterVectorizer and TfidfVectorizer")
    count_config = config['vectorizer']['count']
    tfidf_config = config['vectorizer']['tfidf']
    count_vect = get_vectorizer('count', count_config['max_features'], count_config['max_df'])
    tfidf_vect = get_vectorizer('tfidf', tfidf_config['max_features'], tfidf_config['max_df'])

    results['vectorizer'] = {
        'count_vectorizer': count_vect,
        'tfidf_vectorizer':tfidf_vect
    }

    #Process all the dataset for each candidate and as a whole
    #Process biden dataset
    print("Processing Biden dataset...")
    biden_count_dtm = create_dtm(biden_data, text_column, count_vect)
    biden_tfidf_dtm = create_dtm(biden_data, text_column, tfidf_vect)

    results['biden'] = {
        'count_dtm': biden_count_dtm,
        'tfidf_dtm': biden_tfidf_dtm
    }

    # Process Trump dataset
    print("Processing Trump dataset...")
    trump_count_dtm = create_dtm(trump_data, text_column, count_vect)
    trump_tfidf_dtm = create_dtm(trump_data, text_column, tfidf_vect)

    results['trump'] = {
        'count_dtm': trump_count_dtm,
        'tfidf_dtm': trump_tfidf_dtm
    }

    #Process the combine dataset
    print("Processing combine dataset with duplicates...")
    combine_count_dtm = create_dtm(combine_dups, text_column, count_vect)
    combine_tfidf_dtm = create_dtm(combine_dups, text_column, tfidf_vect)

    results['combine'] = {
        'count_dtm': combine_count_dtm,
        'tfidf_dtm': combine_tfidf_dtm
    }

    print("Topic Modeling Preprocessing Complete!")

    return results

if __name__ == '__main__':
    config_path = 'conf/config.yaml'
    results = topic_preprocess(config_path)


