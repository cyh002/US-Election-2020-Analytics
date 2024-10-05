# Election Tweet Sentiment Analysis Tool

This project aims to develop a cutting-edge tool that accurately measures public sentiment expressed in tweets related to the presidential election. It utilizes advanced Natural Language Processing (NLP) techniques, incorporating modern slang, sarcasm detection, emoji analysis, and real-time sentiment tracking.

## Key Features

- **Real-Time Sentiment Analysis**: Analyze tweets related to the presidential election in real-time.
- **Advanced NLP**: Utilize the DistilBERT model for deeper understanding of tweet language, including slang, sarcasm, and emojis.
- **Visualizations**: Graphical representations of public sentiment using `Streamlit` and `Graphviz`.
- **FastAPI Batch Inference**: Efficient sentiment analysis on batches of tweets using `FastAPI`.
- **Interactive Dashboards**: Create interactive data visualizations to explore election-related public sentiment.

## Tech Stack

- **Natural Language Processing (NLP)**: `Hugging Face Transformers` (DistilBERT)
- **Web Scraping & Tweet Collection**: `Tweepy`
- **Real-Time Dashboards**: `Streamlit`
- **Back-End API**: `FastAPI`
- **Visualization**: `Graphviz`, `Matplotlib`, `Seaborn`, `Plotly`
- **Data Handling**: `Pandas`, `NumPy`
- **Machine Learning**: `PyTorch`

## Project Structure

```plaintext
├── conf/                # Configuration files (e.g., API keys, environment settings)
├── data/                # Data collection and preprocessing
├── src/                 # Source code for training and analysis
│   ├── dataloader.py    # Script to handle data loading
│   ├── graphrag.py      # Visualization script using graphing tools
│   ├── streamlit.py     # Streamlit dashboard app
│   └── train_model.py   # Model training script
├── .gitignore           # Git ignore file to exclude unnecessary files
├── README.md            # Project documentation
├── requirements.txt     # Dependencies for the project
└── US-Election-2020-Analytics.ipynb  # Jupyter notebook for election analytics
```

## Getting Started

### Prerequisites

Make sure you have Python 3.8 or above installed.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/election-tweet-sentiment-analysis.git
   ```
   
2. Navigate into the project directory:
   ```bash
   cd election-tweet-sentiment-analysis
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. **Collect Tweets**: Use the `Tweepy` integration to collect real-time tweets related to the election.
   
2. **Run Streamlit App**: Launch the Streamlit app to explore real-time sentiment analysis:
   ```bash
   streamlit run src/streamlit.py
   ```

3. **Serve API for Batch Inference**: Run the FastAPI server for batch sentiment inference:
   ```bash
   uvicorn src.inference:app --reload
   ```

4. **Generate Visualizations**: Use `graphrag.py` to create sentiment graphs:
   ```bash
   python src/graphrag.py
   ```

### Model Training

To train the sentiment analysis model, use the provided training scripts:

```bash
python src/train_model.py --data data/election_tweets.csv
```

### Visualization

To generate visualizations of sentiment trends, you can use `Streamlit` or `graphrag.py`. Here's how to visualize sentiment trends over time:

```bash
streamlit run src/visualize.py
```

### Inference

For batch inference on collected tweets, use the FastAPI batch processing system:

```bash
curl -X POST "http://127.0.0.1:8000/inference" -H "Content-Type: application/json" -d '{"tweets": ["I love this candidate!", "I hate this policy."]}'
```

## Testing

To run the tests, use `pytest`:

```bash
pytest
```

## Future Enhancements

- **Improved Sarcasm Detection**: Incorporating more complex models for sarcasm and irony detection.
- **Broader Language Support**: Expand to support multiple languages and dialects.
- **Advanced Contextual Analysis**: Analyze sentiment across entire conversation threads for better context.
