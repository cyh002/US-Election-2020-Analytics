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
from IPython.display import display as ipy_display, HTML
import seaborn as sns
import umap.umap_ as umap

def stratified_sample_dtm(dtm: csr_matrix, sample_size, stratify_bins=5, random_seed=42):
    """
    Perform stratified sampling on a large sparse document-term matrix (DTM) 
    based on document length bins (number of non-zero terms per document).

    Args:
        dtm (csr_matrix): A sparse matrix where rows are documents and columns are terms.
        stratify_bins (int): Number of bins to create for stratified sampling (default is 5).
        sample_size (int): Total number of documents to sample (must be <= total number of documents).
        random_seed (int): Seed for reproducibility.

    Returns:
        sample_dtm (csr_matrix): A stratified sample of the original DTM in sparse format.
    """
    # Set random seed for reproducibility- fix the random seed so that the same stratified sample will be chosen
    np.random.seed(random_seed)

    #Make sure that the dtm is a csr_matrix
    if not isinstance(dtm, csr_matrix):
        dtm = csr_matrix(dtm)  # Convert to sparse format

    # Calculate the number of non-zero terms per document (document length)
    doc_lengths = np.array(dtm.getnnz(axis=1))

    # Create bins based on document lengths
    bins = pd.qcut(doc_lengths, q=stratify_bins, duplicates='drop', labels=False)

    # Store indices of sampled documents
    sampled_indices = []

    # Calculate how many samples to draw from each bin
    bin_counts = np.bincount(bins)  # Number of documents in each bin
    bin_sample_sizes = (bin_counts / bin_counts.sum() * sample_size).astype(int)

    # Stratified sampling within each bin
    for bin_id, bin_size in enumerate(bin_sample_sizes):
        bin_indices = np.where(bins == bin_id)[0]  # Indices of documents in this bin
        bin_sample = np.random.choice(bin_indices, size=bin_size, replace=False)  # Sample within bin
        sampled_indices.extend(bin_sample)

    # Ensure sampled_indices is the desired sample size (in case rounding caused off-by-1 error)
    if len(sampled_indices) < sample_size:
        extra_needed = sample_size - len(sampled_indices)
        remaining_indices = np.setdiff1d(np.arange(dtm.shape[0]), sampled_indices)
        extra_sample = np.random.choice(remaining_indices, size=extra_needed, replace=False)
        sampled_indices.extend(extra_sample)

    # Convert sampled indices to a sorted array for efficient indexing
    sampled_indices = np.sort(sampled_indices)

    # Extract the sampled DTM in sparse format
    sample_dtm = dtm[sampled_indices, :]

    return sample_dtm

def lda_model(sample_dtm:pd.DataFrame, topics:int, topic_word_prior:float, doc_topic_prior:float, random_state:int):
    '''
    Create the LDA model and train it.
    
    Args: 
        sample_dtm (pd.DataFrame): The sample of the original document-term matrix (DTM) for training.
        topics (int): The number of topics to find.
        topic_word_prior (float): Prior for topic-word distribution (beta).
        doc_topic_prior (float): Prior for document-topic distribution (alpha).
        random_state (int): Random seed for reproducibility (default is 123).

    Returns:
        lda (LatentDirichletAllocation): A trained LDA model from sklearn that contains the topic-word distributions and other attributes.
    '''
    #State explicitly and change it to int
    topics = int(topics)

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
        sample_dtm (pd.DataFrame): The document-term matrix, where rows represent documents and columns represent terms.
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

def train_and_evaluate(model, params: dict, train_sample, vectorizer, top_words: int, random_state:int) -> pd.DataFrame:
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

def top_words_from_topic(component: np.array, vocab: np.array, topn:int = 10) -> list:
    """
    Extract the top words from each topic based on their importance scores.

    Args: 
        component (np.array): The topic-word matrix from the LDA model, where 
                              each row corresponds to a topic and each column 
                              represents a word's importance in that topic.
        vocab (np.array): The array of vocabulary words corresponding to the columns of the topic-word matrix.
        topn (int): The number of top words to extract for each topic (default is 10).

    Returns:
        list: A list where each element contains the top words for a topic.

    """
    top_words_per_topic = []

    for top in range(component.shape[0]):
        top_words = component[top,:].argsort()[-topn:][::-1].tolist()
        words = vocab[top_words]
        top_words_per_topic.append(words)
        print(f"Topic {top} words: {'|'.join(words)}")
    return top_words_per_topic

def display_pyldavis(display):
    """
    Render a pyLDAvis visualization in a Jupyter notebook.

    Args:
        display: The prepared pyLDAvis visualization object.

    This function converts the pyLDAvis visualization into HTML format and displays it using IPython's display tools. It provides a seamless way to 
    view interactive topic model visualizations within Jupyter notebooks. The visualization offers insights into topic distributions, word relevance, 
    and relationships between topics in the trained model.
    """
    html = pyLDAvis.prepared_data_to_html(display)  # Convert to HTML
    ipy_display(HTML(html))
    # pyLDAvis.enable_notebook()
    pyLDAvis.display(display)

def jaccard_similarity(topic_a: np.array, topic_b: np.array)-> float:
    """
    Calculate the Jaccard similarity between two sets of topics.

    The Jaccard similarity is a measure of similarity between two sets, 
    defined as the size of the intersection divided by the size of the union 
    of the sets.

    Args:
        topic_a (array-like): The top words from topic A.
        topic_b (array-like): The top words from topic B.

    Returns:
        float: The Jaccard similarity value, ranging from 0 to 1.
               A value of 1 indicates identical sets, while 0 indicates no overlap.
    """
    intersection = len(set(topic_a).intersection(set(topic_b)))
    union = len(set(topic_a).union(set(topic_b)))
    return intersection / union

def similarity_heatmap(topic_words_a: np.array, topic_words_b: np.array):
    '''
    Plot a heatmap showing the Jaccard similarity between topics from two different models.

    This function calculates the Jaccard similarity for each pair of topics from two sets of topics 
    (e.g., from two different models). It visualizes the similarities as a heatmap, where each cell 
    represents the Jaccard similarity between a topic from Model A and a topic from Model B.

    Args:
        topic_words_a (list of array-like): A list of word sets representing the topics from Model A.
        topic_words_b (list of array-like): A list of word sets representing the topics from Model B.

    Returns:
        None: Displays a heatmap of Jaccard similarities between the topics of the two models.
    
    Example:
        If Model A has 3 topics and Model B has 2 topics, the heatmap will be a 3x2 grid, where 
        each cell (i, j) shows the Jaccard similarity between the i-th topic of Model A and the 
        j-th topic of Model B.
    '''
    num_topics_a = len(topic_words_a)
    num_topics_b = len(topic_words_b)
    overlap_matrix = np.zeros((num_topics_a, num_topics_b))

    # Populate the overlap matrix
    for i in range(num_topics_a):
        for j in range(num_topics_b):
            overlap_matrix[i, j] = jaccard_similarity(topic_words_a[i], topic_words_b[j])

    # Plot the heatmap of the overlap matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(overlap_matrix, annot=True, cmap="coolwarm", cbar=True)
    plt.title("Top Words Overlap Between Model A and Model B Topics")
    plt.xlabel("Model B Topics")
    plt.ylabel("Model A Topics")
    plt.show()

def document_topic_matrix(model: object, dtm: csr_matrix) -> np.array:
    '''
    Generate the document-topic matrix using a topic modeling algorithm.

    This function applies a topic modeling algorithm (such as LDA, NMF, or similar) to 
    a document-term matrix (DTM) and returns the resulting document-topic matrix. Each row 
    of the document-topic matrix represents a document, and each column represents a topic, 
    with values indicating the relevance or contribution of a particular topic to that document.

    Args:
        model (object): A fitted topic modeling algorithm that implements the `fit_transform` method, 
                        such as Latent Dirichlet Allocation (LDA) or Non-negative Matrix Factorization (NMF).
        dtm (csr_matrix): The document-term matrix where rows correspond to documents and columns correspond to terms (words).

    Returns:
        array-like: A document-topic matrix, where each entry (i, j) represents the weight or relevance of topic j in document i.
    '''
    doc_topic_matrix = model.fit_transform(dtm)
    return doc_topic_matrix

def umap_visualization(doc_topic_matrix: np.arary, n_components:int=2, random_state:int=42):
    '''
    Visualize documents in a lower-dimensional space using UMAP and highlight dominant topics.

    This function applies **UMAP (Uniform Manifold Approximation and Projection)** to the 
    document-topic matrix to reduce its dimensionality for visualization. It plots the documents 
    in a 2D or 3D space (based on the `n_components` parameter) and colors them according to their 
    dominant topic, helping to visually inspect topic clusters.

    Args:
        doc_topic_matrix (array-like): The document-topic matrix, where each row represents 
                                       a document, and each column represents a topic.
        n_components (int): The number of components for UMAP projection (default is 2).
                                      Typically, 2 or 3 components are used for visualization.
        random_state (int): A seed for reproducibility of UMAP results (default is 42).

    Returns:
        None: Displays a scatter plot of the UMAP-transformed documents, colored by their dominant topics.
    '''
    # Apply UMAP
    umap_model = umap.UMAP(n_components=n_components, random_state= random_state, n_jobs =1)
    umap_results = umap_model.fit_transform(doc_topic_matrix)

    # Find the dominant topics
    dominant_topics = np.argmax(doc_topic_matrix, axis=1)

    #Get the number of unqiue topics
    num_topics = len(np.unique(dominant_topics))

    #Get the color map
    cmap = plt.get_cmap('tab10', num_topics)

    # Plot the UMAP-transformed documents with a discrete colormap
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(umap_results[:, 0], umap_results[:, 1], c=dominant_topics, cmap='tab10', s=10, alpha=0.7)

    # Create a legend by plotting invisible points with appropriate colors
    for topic in range(num_topics):
        plt.scatter([], [], color=cmap(topic), label=f'Topic {topic}')

    # Add the legend to the plot
    plt.legend(title='Dominant Topic')

    # Set plot labels and title
    plt.title("UMAP Visualization of Documents")
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")

    # Show the plot
    plt.show()

def training_pipeline(config_path: str, size, dtm=None, col_name: str = 'no_stopwords', vectorizer=None, random_seed: int = 0, sample_size: int = 5000, topn: int = 10, random_state: int = 123):
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

    # # Generate a small sample of the DTM for hyperparameter tuning
    print("Generating a small training sample...")
    sample_dtm = stratified_sample_dtm(dtm, sample_size=5000)

    # Ensure the DTM is in CSR matrix format
    if not isinstance(dtm, csr_matrix):
        print("Converting DTM to csr_matrix...")
        dtm = csr_matrix(dtm)  # Convert to sparse format

    # Set a seed for reproducibility of random seed generation
    np.random.seed(42)  

    # Create random seeds
    random_seeds = np.random.randint(1, 10000, size=size)  # Fixed set of seeds
    print(f"Generated random seeds: {random_seeds}")

    #  Prepare for multiple seed runs
    print("Running LDA with different seeds...")
    results = []  # Store models, displays, scores, and seeds
    optimal_topic_list = [] #Store the optimal number of topic per run

    for seed in random_seeds:
        print(f"Training LDA with seed {seed}...")

        # Train and evaluate on a single seed
        records = train_and_evaluate(lda_model, lda_params, sample_dtm, vectorizer,top_words=topn, random_state=seed)
        
        # Get the best parameters and their corresponding score for this seed
        best_params, best_score = get_best_score(records)

        # Retrain the model on the entire dataset using the best parameters
        print(f"Retraining with best parameters for seed {seed}...")
        best_model = lda_model(dtm, **best_params, random_state=seed)
        vocab = vectorizer.get_feature_names_out()
        best_component = best_model.components_
        optimal_topics = best_model.components_.shape[0]
        optimal_topic_list.append(optimal_topics)

        #Calculate the mean umass score for the best model
        print(f"Calculating mean UMass coherence score for seed {seed}...")
        mean_umass_score = mean_umass(topn, dtm, best_component, vocab)

        # Create pyLDAvis visualization
        print(f"Generating pyLDAvis for seed {seed}...")
        lda_display = pyLDAvis.lda_model.prepare(best_model, dtm, vectorizer)

        # Store results for this seed
        results.append((best_model, lda_display, mean_umass_score, seed, best_component))

    # Sort the results by score (highest first)
    print("Sorting models by score...")
    sorted_results = sorted(results, key=lambda x: x[2], reverse=True)

    print("Training process is completed!")

    return sorted_results, optimal_topic_list

if __name__ == '__main__':
    config_path = 'conf/train_model.yaml'
    sorted_results = training_pipeline(config_path)

