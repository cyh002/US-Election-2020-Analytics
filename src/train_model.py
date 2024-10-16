#Import all the necessary libraries
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer
from tmtoolkit.topicmod.evaluate import metric_coherence_gensim
import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import yaml
from preprocessing import load_config
import pyLDAvis
import pyLDAvis.gensim
import pyLDAvis.lda_model
from scipy.sparse import csr_matrix

def random_sample(dtm: pd.DataFrame, random_seed: int, sample_size:int) -> pd.DataFrame:
    '''
    Randomly sample a subset of rows from a document-term matrix (DTM) based on the specified sample size and random seed.

    Args:
        dtm (pd.DataFrame): A document-term matrix where rows correspond to documents and columns represent terms.
        random_seed (int): Used to ensure reproducibility of the random sampling process.
        sample_size (int): The number of rows (documents) to sample from the DTM. This must be less than the total number of rows in the DTM.

    Returns:
        sample_dtm (pd.DataFrame): A sparse matrix containing the randomly sampled subset of rows from the original DTM.
    '''
    #Ensure that the sample size is within the dtm shape
    assert sample_size < dtm.shape[0], "Sample size exceeds the number of rows in the full DTM."
    #Insert random seed
    np.random.seed(random_seed)
    # Sample rows directly from the sparse matrix without converting the entire matrix to dense format
    sample_indices = np.random.choice(dtm.shape[0], sample_size, replace=False)
    #Create the sample dtm
    sample_dtm = pd.DataFrame(dtm.iloc[sample_indices])
    return sample_dtm

def lda_model(sample_dtm:pd.DataFrame, topics:int, topic_word_prior:float, doc_topic_prior:float, random_state:int = 123):
    '''
    Create the LDA model and train it.
    
    Args: 
        sample_dtm (csr_matrix): The sample of the original document-term matrix (DTM) for training, in sparse format (CSR matrix).
        topics (int): The number of topics to find.
        topic_word_prior (float): Prior for topic-word distribution (beta).
        doc_topic_prior (float): Prior for document-topic distribution (alpha).
        random_state (int): Random seed for reproducibility (default is 123).

    Returns:
        lda (LatentDirichletAllocation): A trained LDA model from sklearn that contains the topic-word distributions and other attributes.
    '''
    # Fit the LDA model
    lda = LDA(n_components=topics,
              topic_word_prior=topic_word_prior,
              doc_topic_prior=doc_topic_prior,
              n_jobs=-1,
              random_state=random_state)
    
    # Train the model on the CSR matrix
    lda.fit(sample_dtm)
    
    return lda

def mean_umass(top_number_words: int, sample_dtm: pd.DataFrame, lda_component: np.ndarray, vocab: np.ndarray) -> float:
    '''
    Compute the UMass coherence score for the given topic-word distributions using the provided document-term matrix (DTM).
    This metric helps evaluate how coherent the topics are by calculating how often the top words for each topic co-occur in the documents.
    
    Args:
        top_number_words (int): The number of top words per topic to consider when computing coherence.
        sample_dtm (csr_matrix or np.ndarray): The document-term matrix, where rows represent documents and columns represent terms.
            If passed as a sparse matrix (csr_matrix), it will be converted to a dense format (numpy array).
        lda_component (np.ndarray): The topic-word distribution matrix from the LDA model. 
            Shape is (n_topics, n_words), where each row represents a topic, and each column represents a word's importance in that topic.
        vocab (np.ndarray): The vocabulary array representing the words used in the vectorizer. Each entry corresponds to a word from the DTM.
    
    Returns:
        mean_umass (float): The mean UMass coherence score across all topics. A score closer to 0 indicates better coherence (i.e., the top words in each topic co-occur frequently in the documents).
    '''
    # Convert the sparse matrix to a dense format if necessary
    if isinstance(sample_dtm, csr_matrix):
        sample_dtm = sample_dtm.toarray()  # Convert to dense matrix (numpy array)
    
    # Calculate UMass coherence
    umass = metric_coherence_gensim(measure="u_mass", 
                                    top_n=top_number_words,
                                    topic_word_distrib=lda_component,
                                    dtm=sample_dtm,
                                    vocab=vocab)
    # Find the mean of the coherence of the top words
    mean_umass = np.mean(umass)
    return mean_umass

def train_and_evaluate(model, params: dict, train_sample, vectorizer, top_words: int, random_state:int = 123) -> pd.DataFrame:
    '''
    Train and evaluate the LDA model with different hyperparameter combinations and calculate the UMass coherence score for each.

    Args: 
        model (function): A function to create and train the LDA model. This function should accept the `train_sample` and hyperparameters from `params`.
        params (dict): A dictionary containing the hyperparameters to test. Keys are hyperparameter names (e.g., 'topics', 'topic_word_prior'),
                       and values are lists of possible values for those hyperparameters. These will be combined and evaluated.
        train_sample (csr_matrix or pd.DataFrame): A document-term matrix (DTM) sample used to train the LDA model. It can be either in sparse (csr_matrix) or dense (DataFrame) format.
        vectorizer (CountVectorizer or TfidfVectorizer): A vectorizer object that has been fit to the dataset and used to generate the document-term matrix.
        top_words (int): The number of top words to use when calculating the UMass coherence score for each topic.
        random_state (int): Random seed for reproducibility (default is 123).

    Returns:
        pd.DataFrame: A DataFrame containing the parameter combinations and their corresponding UMass coherence scores. 
                      Each row represents a different parameter combination, and the columns store the parameter values and their mean UMass score.
    '''
    # Extract the vocabulary from the vectorizer
    vocab = vectorizer.get_feature_names_out()

    # Get the hyperparameter names and values from the params dictionary
    param_names = params.keys()
    param_values = params.values()

    # Generate all possible combinations of the hyperparameter values
    param_combinations = list(product(*param_values))

    # List to store the results
    records = []

    for param_combination in param_combinations:
        # Create a dictionary of the current hyperparameter combination
        current_params = dict(zip(param_names, param_combination))
        
        # Initialize and train the model using the current parameters
        lda = model(train_sample, **current_params, random_state=random_state)

        # Get the topic-word distribution matrix (lda.components_)
        lda_component = lda.components_

        # Calculate the UMass coherence score for the current model
        score = mean_umass(top_words, train_sample, lda_component, vocab)

        # Store the parameter combination and its corresponding score
        record = {
            **current_params,
            "mean_umass": score
        }
        records.append(record)

    # Convert the results to a DataFrame and return
    df = pd.DataFrame(records)

    return df

def get_best_score(df: pd.DataFrame):
    '''
    Function to find the best mean umass score and get its coresponding parameters

    Args:
        df (pd.DataFrame): the dataframe of results from the training

    Return: 
        params(dict):{params:float} A dcitionary consist of the name and the value of the parameter
        score(float): the best umean score
    '''
    # Find the row with the score closest to 0
    closest_row = df.loc[df['mean_umass'].abs().idxmin()]
    # Extract the parameters
    params = closest_row.drop('mean_umass').to_dict()
    score = closest_row['mean_umass']
    return params, score

def view_best_score(df:pd.DataFrame, x_label:str ='Number of Topics', y_label:str ='Mean UMass Score', title:str ='UMass Coherence Scores'):
    '''
    Function to create a graph of the UMass score against topics to find the best number of topics.

    Args:
        df (pd.DataFrame): DataFrame containing the results of topic modeling experiments. Must include columns 
                           'topic_word_prior', 'doc_topic_prior', 'topics', and 'mean_umass'.
        x_label (str): Label for the x-axis (default is 'Number of Topics').
        y_label (str): Label for the y-axis (default is 'Mean UMass Score').
        title (str): Title for the plot (default is 'UMass Coherence Scores').

    Returns:
        None. Displays a plot.
    '''
    plt.figure(figsize=(10, 6))
    
    # Loop through unique combinations of topic_word_prior and doc_topic_prior
    for twp in df['topic_word_prior'].unique():
        for dtp in df['doc_topic_prior'].unique():
            subset = df[(df['topic_word_prior'] == twp) & (df['doc_topic_prior'] == dtp)]
            plt.plot(subset['topics'], subset['mean_umass'], label=f"topic_word_prior={twp}, doc_topic_prior={dtp}")

    # Add labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    
    # Add legend and grid
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

def training_pipeline(config_path: str, dtm=None, col_name: str = 'no_stopwords', vectorizer=None, random_seed: int = 0, sample_size: int = 5000, topn: int = 10, random_state: int = 123):
    '''
    Training pipeline to find the best number of topics, extract top words, and visualize them using pyLDAvis.

    Args:
        config_path (str): Path to the configuration file (YAML).
        dtm (csr_matrix or pd.DataFrame): The document-term matrix. Can be in sparse or dense format.
        col_name (str): The column name containing the cleaned text data (default is 'no_stopwords').
        vectorizer (CountVectorizer or TfidfVectorizer): The vectorizer used to transform the text data.
        random_seed (int): Random seed for reproducibility (default is 0).
        sample_size (int): Number of documents to sample for the small training set (default is 5000).
        topn (int): Number of top words to extract per topic (default is 10).
        random_state (int): Random seed for reproducibility (default is 123).

    Returns:
        tuple: 
            - results (dict): Top words for each topic based on the best model.
            - lda_display (pyLDAvis object): The pyLDAvis visualization object for the topics.
    '''
    # Load configuration
    print("Loading config...")
    config = load_config(config_path)

    # Load and adjust LDA parameters
    print("Loading parameters from config...")
    lda_params = config['lda_params']
    lda_params['topics'] = list(range(lda_params['topics']['start'],
                                      lda_params['topics']['end'] + 1,
                                      lda_params['topics']['step']))

    # Generate a small sample of the DTM for hyperparameter tuning
    print("Generating a small training sample...")
    sample_dtm = random_sample(dtm, random_seed, sample_size)

    # Train and evaluate the model on the sample data
    print("Training and evaluating on the small sample size...")
    records = train_and_evaluate(lda_model, lda_params, sample_dtm, vectorizer, top_words=topn, random_state=random_state)

    # Plot the results of training
    print("Plotting results...")
    view_best_score(records)

    # Get the best parameters and their corresponding score
    print("Extracting the best parameters and score...")
    best_params, best_score = get_best_score(records)

    # Retrain the model on the entire dataset with the best parameters
    print("Retraining with the best parameters on the full dataset...")
    best_params['topics'] = int(best_params['topics'])  # Ensure 'topics' is an integer
    best_model = lda_model(dtm, **best_params, random_state=random_state)  # Removed 'vectorizer' as it's not used in lda_model
    vocab = vectorizer.get_feature_names_out()

    # Check if the document-term matrix (dtm) is a CSR matrix, if not convert it
    if not isinstance(dtm, csr_matrix):
        print("Converting dtm to csr_matrix...")
        dtm = csr_matrix(dtm)

    # Create pyLDAvis visualization
    print("Generating the pyLDAvis display...")
    lda_display = pyLDAvis.lda_model.prepare(best_model, dtm, vectorizer)

    print("Training process is completed!")

    return lda_display

if __name__ == '__main__':
    config_path = 'conf/train_model.yaml'
    results = training_pipeline(config_path)

