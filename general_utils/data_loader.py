import pandas as pd
from src.preprocessing import load_config, cast_data_type
from src.misc_utils import engagement_score, normalization, normalize_scores
import dotenv
import os

class DataLoader:
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.data = None
        self.load_and_process_data()

    @property
    def openai_api_key(self):
        dotenv.load_dotenv()
        return os.getenv('openai_api_key')

    @property
    def openai_base_url(self):
        return self.config['openai']['base_url']

    @property
    def openai_model(self):
        return self.config['openai']['model']

    @property
    def geojson_url(self):
        return self.config['streamlit']['geojson_url']

    @property
    def llm_analyzer_count(self):
        return self.config['streamlit']['llm_analyzer_count']

    def load_and_process_data(self):
        data = pd.read_csv(self.config['streamlit']['data'])
        data = cast_data_type(data)
        data['engagement'] = engagement_score(
            data['likes'],
            data['retweet_count'],
            data['user_followers_count']
        )
        data['normalized_score'] = normalization(
            data['engagement'],
            data['sentiment'],
            data['confidence']
        )
        
        data['normalized_score'] = normalize_scores(data['normalized_score'])
        data = cast_data_type(data)
        self.data = data

    def get_data(self):
        return self.data