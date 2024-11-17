# Twitter Sentiment Analysis: 2020 US Election

An interactive dashboard analyzing Twitter sentiment during the 2020 US Presidential Election, focusing on tweets related to Donald Trump and Joe Biden.
> **Data Source**: [US Election 2020 Tweets🌟](https://www.kaggle.com/datasets/manchunhui/us-election-2020-tweets/)
> **Live Demo**: [Streamlit Cloud: Twitter Sentiment Analysis Dashboard 😉](https://us-election-2020-analytics.streamlit.app/) 
> **Heroku Deployment**: [Backup: Twitter Sentiment Analysis Dashboard 🤯](https://us-election-analytics-2020-4ee3c0a80b5f.herokuapp.com/)
## 📊 Features

- **Data Overview**: Explore the raw data and its structure
- **User Analysis**: Analyze user influence and detect suspicious accounts
- **Daily Analysis**: AI-powered daily sentiment summaries using DeepSeek-Chat
- **Choropleth Map**: Geographic sentiment distribution across US states
- **Sentiment Distribution**: Compare sentiment patterns between candidates
- **Time Series Analysis**: Track sentiment trends over time
- **Word Cloud**: Visualize frequently discussed topics

## 🛠️ Technologies

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **NLP & Sentiment Analysis**: 
  - RoBERTa (CardiffNLP/twitter-xlm-roberta-base-sentiment)
  - DeepSeek-Chat LLM
- **Data Validation**: Pydantic
- **Testing**: Python unittest

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/twitter-sentiment-analysis.git
cd twitter-sentiment-analysis
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Set up environment variables
```bash
export openai_api_key=your_api_key
```

4. Run the application
```bash
streamlit run app/streamlit_app.py
```

## ⚙️ Configuration

The project uses YAML configuration files (`conf/config.yaml`) for managing:

### Data Paths
```yaml
data:
  trump_path: 'data/raw/hashtag_donaldtrump.csv'
  biden_path: 'data/raw/hashtag_joebiden.csv'
  output_path: 'data/cleaned/english_tweets.csv'
  results_path: 'data/results/'
```

### Processing Parameters
```yaml
vectorizer:
  count:
    type: "count"
    max_features: 1000
    max_df: 0.5
```

### Application Settings
```yaml
streamlit:
  port: 8501
  data: 'data/results/results_xlm.csv'
  llm_analyzer_count: 100
```

### LLM Configuration
```yaml
openai:
  base_url: 'https://api.deepseek.com'
  model: 'deepseek-chat'
```

## 🗂️ Project Structure

```
├── app/                      # Main application directory
│   ├── general_utils/       # Utility functions
│   ├── pages/              # Streamlit pages
│   └── streamlit_app.py    # Main app file
├── conf/                    # Configuration files
├── data/                    # Data directory
│   ├── cleaned/            # Processed data
│   ├── raw/                # Raw tweet data
│   └── results/            # Analysis results
├── src/                    # Source code
│   ├── preprocessing.py    # Data preprocessing
│   ├── topic_modeling.py   # Topic analysis
│   └── train_model.py      # Model training
└── tests/                  # Test files
```

## 🐳 Docker & Heroku Deployment

### Docker Build

1. Build the Docker image
```bash
# Build with OpenAI API key
docker build -f docker/Dockerfile . -t streamlit-app:latest
```

2. Test locally
```bash
# Run container with port mapping
docker run -d -p 8501:8501 -e PORT=8501 streamlit-app:latest

# View at http://localhost:8501
```

### Heroku Deployment

1. Install Heroku CLI and login
```bash
curl https://cli-assets.heroku.com/install.sh | sh
heroku login
heroku container:login
```

2. Create Heroku app
```bash
heroku create us-election-analytics-2020
```

3. Set environment variables
```bash
heroku config:set openai_api_key=your_key_here -a us-election-analytics-2020
```

4. Deploy container
```bash
# Tag image for Heroku
docker tag streamlit-app:latest registry.heroku.com/us-election-analytics-2020/web

# Push to Heroku registry
docker push registry.heroku.com/us-election-analytics-2020/web

# Release the container
heroku container:release web -a us-election-analytics-2020
```

5. Open the app
```bash
heroku open -a us-election-analytics-2020
```

### Troubleshooting

- View logs: `heroku logs --tail -a us-election-analytics-2020`
- Restart app: `heroku restart -a us-election-analytics-2020`
- Check build: `heroku builds -a us-election-analytics-2020`


## 👥 Team Members

<div align="center">
<table>
<tr>
    <td align="center">
        <a href="https://www.linkedin.com/in/christopher-chi-yang-hoo-570698bb/">
            <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="Christopher LinkedIn"/>
            <br />
            <b>Christopher Chi Yang Hoo</b>
        </a>
        <br />
        Data Scientist & ML Engineer
    </td>
    <td align="center">
        <a href="mailto:kwong.victoriaa@gmail.com">
            <img src="https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white" alt="Victoria Email"/>
            <br />
            <b>Victoria Kwong Jia Ying</b>
        </a>
        <br />
        Data Scientist & ML Engineer
    </td>
    <td align="center">
        <a href="https://www.linkedin.com/in/anthony-kwa/">
            <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="Anthony LinkedIn"/>
            <br />
            <b>Anthony Kwa</b>
        </a>
        <br />
        Data Scientist & ML Engineer
    </td>
</tr>
</table>
</div>

### 🎓 Project Context
This project was developed as part of the Data and Visual Analytics course at Georgia Tech. Our team collaborated to create a comprehensive sentiment analysis dashboard for analyzing Twitter discussions during the 2020 US Presidential Election.

### 👨‍💻 Role Distribution
- **Christopher Chi Yang Hoo**: Led backend development, implemented LLM integration
- **Victoria Kwong Jia Ying**: Managed data preprocessing, statistical analysis, reporting
- **Anthony Kwa**: Managed data preprocessing, statistical analysis, reporting

## 📚 Documentation

For detailed information about metrics and calculations, visit the Glossary page in the application.

