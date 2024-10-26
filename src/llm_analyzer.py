import os
from typing import List, Optional
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langchain.chains.openai_functions.base import create_structured_output_chain
from pydantic import BaseModel, Field
import pandas as pd
# Define the data models
class CandidateSentiment(BaseModel):
    candidate: str = Field(description="Name of the political candidate, e.g., Biden, Trump")
    sentiment: str = Field(description="Sentiment towards the candidate (positive/negative/neutral)")
    score: float = Field(description="Sentiment score between -1 and 1")
    key_topics: List[str] = Field(description="List of key topics associated with the candidate")

class OverallAnalysis(BaseModel):
    comparison: str = Field(description="Comparative analysis between candidates")
    confidence_score: float = Field(description="Confidence score of the analysis between 0 and 1")

class DailyReport(BaseModel):
    sentiments: List[CandidateSentiment] = Field(description="List of sentiment analysis for each candidate")
    overall_analysis: OverallAnalysis = Field(description="Overall comparative analysis between candidates")
    additional_insights: Optional[str] = Field(description="Additional political insights from the analysis")

# LLM Analyzer class
class LLMAnalyzer:
    def __init__(
        self,
        openai_api_key: str,
        model: str = "deepseek-chat",
        base_url="https://api.deepseek.com",
        temperature: float = 1.0
    ):
        print("Initializing LLMAnalyzer")
        self.llm = ChatOpenAI(model=model, temperature=temperature, api_key=openai_api_key, base_url=base_url)
        self.parser = PydanticOutputParser(pydantic_object=DailyReport)
        self.prompt = self._create_prompt()
    
    def test_model(self):
        print("Testing model connection...")
        response = self.llm.invoke("Hello, how are you?")
        print(f"Response: {response}")
        return response

    def _create_prompt(self):
        print("Creating prompt template...")
        template = """
        You are an expert data analyst and political analyst.
        
        Analyze the following tweet data and generate a structured report.
        The data consists of cleaned tweets and their engagement metrics for the latest day.
        The columns are:
        - clean_tweet: Cleaned tweet text
        - engagement: Scaled engagement score considering likes and retweets, scaled by the logarithm of follower count.
        
        Data:
        {data}
        
        Based on the tweets:
        1. Analyze sentiment and key topics for both Biden and Trump
        2. Provide sentiment scores between -1 (most negative) and 1 (most positive)
        3. List key topics discussed in relation to each candidate
        4. Compare the candidates and provide a confidence score for your analysis
        5. Add any additional insights you observe

        {format_instructions}
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["data"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        return prompt

    def generate_daily_report(self, df: pd.DataFrame) -> Optional[DailyReport]:
        try:
            # Convert DataFrame to a more readable format
            data_str = df[['clean_tweet', 'engagement']].to_string()
            
            print(f"Data to be analyzed: {data_str}")
            
            # Format the prompt with data
            formatted_prompt = self.prompt.format(data=data_str)
            
            # Get response from LLM
            response = self.llm.invoke(formatted_prompt)
            
            # Parse the response into DailyReport structure
            parsed_response = self.parser.parse(response.content)
            print("Parsed response:", parsed_response)
            
            return parsed_response
                
        except Exception as e:
            print(f"Error generating report: {e}")
            return None

    @staticmethod
    def filter_latest_day(df: pd.DataFrame) -> pd.DataFrame:
        latest_df = df[df['created_date'] == df['created_date'].max()].copy()
        print(f"Filtered data for latest day with {len(latest_df)} records")
        return latest_df

# Test the functionality
if __name__ == "__main__":
    # Simulate some data
    data = {
        "clean_tweet": [
            "Biden speaks on climate change in a big way",
            "Trump says economy will be great again",
            "People unhappy with Biden's health care plan",
            "Trump rallies support in the South",
            "Mixed reviews on Biden's student debt relief proposal"
        ],
        "engagement": [100, 150, 120, 80, 95],
        "created_date": ["2023-11-01", "2023-11-01", "2023-11-01", "2023-11-01", "2023-11-01"]
    }
    df = pd.DataFrame(data)
    
    # Ask for OpenAI API key
    print("Please enter your OpenAI API key:")
    openai_api_key = input()

    analyzer = LLMAnalyzer(openai_api_key=openai_api_key)

    # Test model connection
    analyzer.test_model()

    # Filter latest day and analyze
    latest_data_df = analyzer.filter_latest_day(df)
    report = analyzer.generate_daily_report(latest_data_df)
    
    # Output result for verification
    if report:
        print("\nGenerated Daily Report:")
        print(f"Sentiments: {report.sentiments}")
        print(f"Overall Analysis: {report.overall_analysis}")
        print(f"Additional Insights: {report.additional_insights}")