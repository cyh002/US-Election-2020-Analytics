#Import all the libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

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

