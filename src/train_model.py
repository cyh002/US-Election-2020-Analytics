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

def random_sample(dtm: pd.DataFrame, random_seed: int, sample_size:int) -> pd.DataFrame:
    '''
    Randomly sample a subset of rows from a document-term matrix (DTM) based on the specified sample size and random seed.

    Args:
        dtm (pd.DataFrame): A document-term matrix where rows correspond to documents and columns represent terms.
        random_seed (int): Used to ensure reproducibility of the random sampling process.
        sample_size (int): The number of rows (documents) to sample from the DTM. This must be less than the total number of rows in the DTM.

    Returns:
        sample_dtm (pd.DataFrame): A pandas DataFrame containing the randomly sampled subset of rows from the original DTM.
    '''
    #Ensure that the sample size is within the dtm shape
    assert sample_size < dtm.shape[0], "Sample size exceeds the number of rows in the full DTM."
    #Insert random seed
    np.random.seed(random_seed)
    # Sample rows directly from the sparse matrix without converting the entire matrix to dense format
    sample_indices = np.random.choice(dtm.shape[0], sample_size, replace=False)
    # Convert only the sampled rows to a dense DataFrame
    sample_dtm = pd.DataFrame(dtm.iloc[sample_indices])
    return sample_dtm

def lda_model(sample_dtm, vectorizer, topics, topic_word_prior, doc_topic_prior, random_state = 123):
    '''
    Create the LDA model and train it

    Args: 
        sample_dtm (pd.DataFrame): the sample of the original dtm for training
        topics (int): The number of topics to find
        topic_word_prior (float): Prior of topic word distribution beta.
        doc_topic_prior (float): Prior of document topic distribution theta.

    Returns:
        lda_component(np.array): shape:(number of topics, value of each feature)

    '''
       # Fit the LDA model
    lda = LDA(n_components=topics,
                topic_word_prior=topic_word_prior,
                doc_topic_prior=doc_topic_prior,
                n_jobs=-1,
                random_state=random_state)
    # Train the model
    lda.fit(sample_dtm)

    #evaluate the training
    lda_component = lda.components_

    #get the  vocab out
    vocab = vectorizer.get_feature_names_out()
    return lda_component, vocab

def mean_umass(top_number_words, sample_dtm, lda_component, vocab):
    '''
    Evaluation metric to find the best parameters

    Args:
        top_number_words (int): Number of words per topic to evaluate 
        sample_dtm (pd.DataFrame): Sample of the document term matrix
        vocab (object): Words shortlisted by the vectorizer. 

    Returns:
        mean_umass (float): the mean umass between all the words. A value closer to 0 indicate perfect coherence. 
    '''
    umass = metric_coherence_gensim(measure="u_mass", 
                                    top_n=top_number_words,
                                    topic_word_distrib=lda_component,
                                    dtm=sample_dtm,
                                    vocab=vocab)
    #Find the mean of the coherence of the top 5 words
    mean_umass = np.mean(umass)
    return mean_umass

def train_and_evaluate(model, params, train_sample, vectorizer, top_words):
    '''
    For training and evaluating the LDA model

    Args: 
        model: the type of model to run
        params: the parameters to train on. Change on the train_model.yaml
        train_sample (pd.DataFrame): training on a small sample of the original dataframe
        vectorizer (object): Countervectorizer or TfidfVectorizer 
        top_words (int): the number of words to evaluate for the umass

    Return:
        record (pd.DataFrame): a record of the paramters trained on and the mean_umass score
    '''
   # Get all parameter names and values from the param_grid
    param_names = params.keys()
    param_values = params.values()

    # Create a list of all combinations of parameters using itertools.product
    param_combinations = list(product(*param_values))

    records = []

    for param_combination in param_combinations:
        # Create a dictionary of current parameter values
        current_params = dict(zip(param_names, param_combination))
        
        # Print the current parameters being used
        print(f"Running model with parameters: {current_params}")
        
        # Initialize the model with the current parameters
        lda_component, vocab = model(train_sample, vectorizer, **current_params)

        #Evaluate
        score = mean_umass(top_words, train_sample, lda_component, vocab)

        #  Store the results
        record = {
            **current_params,
            "mean_umass": score
        }
        records.append(record)

    df = pd.DataFrame(records)

    return df

def get_best_score(df):
    '''
    Function to find the best mean umass score and get its coresponding parameters

    Args:
        df(pd.DataFrame): the dataframe of results from the trainin

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

def view_best_score(df):
    '''
    Function to create a graph of the umean score against topics to find the best number of topics
    '''
    plt.figure(figsize=(10, 6))
    for twp in df['topic_word_prior'].unique():
        for dtp in df['doc_topic_prior'].unique():
            subset = df[(df['topic_word_prior'] == twp) & (df['doc_topic_prior'] == dtp)]
            plt.plot(subset['topics'], subset['mean_umass'], label=f"topic_word_prior={twp}, doc_topic_prior={dtp}")

    # Add labels and title
    plt.xlabel('Number of Topics')
    plt.ylabel('Mean UMass Score')
    plt.title('UMass Coherence Scores for Different Topic and Prior Combinations')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

def get_top_words(params, lda_component, best_vocab, topn):
    '''
    Function to print out the best words for each topic
    '''
    # Top words for the topic
    for top in range(params['topics']):
        top_words = lda_component[top,:].argsort()[-topn:][::-1].tolist()
        words = best_vocab[top_words]
        print(f"Topic {top} words: {'|'.join(words)}")

    return

def training_pipline(dtm, vectorizer, random_seed, sample_size, topn, params):
    # Reuse the load_config function for the train_model
    config = load_config('conf/train_model.yaml')

    # Access the `lda_params` from the config
    lda_params = config['lda_params']

    # Adjust lda_params: create the 'topics' range using start, end, step
    lda_params['topics'] = list(range(lda_params['topics']['start'],
                                    lda_params['topics']['end'] + 1,
                                    lda_params['topics']['step']))

    #Get a sample of the dtm
    sample_dtm =  random_sample(dtm, random_seed, sample_size)

    #Train and evaluate on a small dataset
    records = train_and_evaluate(lda_model, lda_params, sample_dtm, vectorizer, top_words=10)

    #Plot the graph of the results
    view_best_score(records)

    # Get the best params and score from the records
    best_params, best_score = get_best_score(records)

    #Retrain the model on the whole dataset with the best params
    best_params['topics'] = int(best_params['topics'])
    best_lda_component, vocab = lda_model(dtm, vectorizer,**best_params)

    #Print out the top words for each topic
    results = get_top_words(best_params, best_lda_component, vocab, topn)

    return results, vocab, 



