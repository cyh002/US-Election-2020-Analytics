import pandas as pd
import gc
from functools import lru_cache
from src.preprocessing import load_config, cast_data_type
from src.misc_utils import engagement_score, normalization, normalize_scores
import dotenv
import os

class DataLoader:
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.data = None
        self.chunk_size = 10000
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
    
    @lru_cache(maxsize=1)
    def load_and_process_data(self):
        chunks = []
        
        # Process in chunks
        for chunk in pd.read_csv(self.config['streamlit']['data'], 
                                chunksize=self.chunk_size):
            # Use existing cast_data_type function
            chunk = cast_data_type(chunk)
            
            # Calculate metrics
            chunk['engagement'] = engagement_score(
                chunk['likes'],
                chunk['retweet_count'],
                chunk['user_followers_count']
            )
            
            chunks.append(chunk)
            gc.collect()  # Clear memory after each chunk
            
        # Combine processed chunks
        self.data = pd.concat(chunks)
        
        # Apply final transformations
        self.data['normalized_score'] = normalization(
            self.data['engagement'],
            self.data['sentiment'],
            self.data['confidence']
        )
        self.data['normalized_score'] = normalize_scores(self.data['normalized_score'])
        
        # Final type casting
        self.data = cast_data_type(self.data)
        
        # Cleanup
        del chunks
        gc.collect()
        
    def get_data(self):
        return self.data