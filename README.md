# Twitter Sentiment Analysis: 2020 US Election

An interactive dashboard analyzing Twitter sentiment during the 2020 US Presidential Election, focusing on tweets related to Donald Trump and Joe Biden.

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
export DEEPSEEK_API_KEY=your_api_key
```

4. Run the application
```bash
python -m streamlit run app/streamlit_app.py
```

## 📚 Documentation

For detailed information about metrics and calculations, visit the Glossary page in the application.

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
        <br />
    </td>
    <td align="center">
        <a href="mailto:kwong.victoriaa@gmail.com">
            <img src="https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white" alt="Victoria Email"/>
            <br />
            <b>Victoria Kwong Jia Ying</b>
        </a>
        <br />
        Data Scientist & ML Engineer
        <br />
    </td>
    <td align="center">
        <a href="https://www.linkedin.com/in/anthony-kwa/">
            <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="Anthony LinkedIn"/>
            <br />
            <b>Anthony Kwa</b>
        </a>
        <br />
        Data Scientist & ML Engineer
        <br />
    </td>
</tr>
</table>

</div>

### 🎓 Project Context
This project was developed as part of the Data and Visual Analytics course at Georgia Tech. Our team collaborated to create a comprehensive sentiment analysis dashboard for analyzing Twitter discussions during the 2020 US Presidential Election.

### 👨‍💻 Role Distribution
- **Christopher Chi Yang Hoo**: Led backend development, implemented LLM integration.
- **Victoria Kwong Jia Ying**: Managed data preprocessing, statistical analysis, reporting
- **Anthony Kwa**: Managed data preprocessing, statistical analysis, reporting.
