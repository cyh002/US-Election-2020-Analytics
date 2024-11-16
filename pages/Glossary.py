import streamlit as st

class GlossaryPage:
    def __init__(self):
        self.render_page()
    
    def render_page(self):
        st.header("📚 Glossary and Metrics")
        
        # Sentiment Analysis Section
        self.render_sentiment_section()
        
        # Engagement Metrics Section
        self.render_engagement_section()
        
        # Account Analysis Section
        self.render_account_section()
        
        # Visualization Section
        self.render_visualization_section()
        
        # Data Processing Section
        self.render_data_processing_section()

    def render_sentiment_section(self):
        st.subheader("Sentiment Analysis")
        
        st.markdown("""
        #### RoBERTa Model
        - Uses CardiffNLP Twitter-XLM-RoBERTa model for tweet classification:
            - Positive (1)
            - Neutral (0)
            - Negative (-1)
        - **Confidence Score**: Model's prediction probability (0-1)
        
        #### Daily Sentiment Aggregation (DeepSeek-Chat LLM)
        - Daily sentiment analysis summaries
        - Key topic identification (max 30 topics)
        - Comparative analysis between candidates
        - Additional political insights
        """)

    def render_engagement_section(self):
        st.subheader("Engagement Metrics")
        
        with st.expander("Base Engagement Score"):
            st.code("""
            engagement = likes + retweets
            engagement_rate = (engagement / followers) * 100
            scaled_engagement = engagement_rate * log(followers)
            """)
            st.markdown("""
            - Combines likes and retweets
            - Normalized by follower count
            - Logarithmically scaled by followers to balance influence
            """)
            
        with st.expander("Normalized Score"):
            st.code("""
            adjusted_sentiment = (sentiment + 1) / 2  # Transform to [0,1]
            normalized = engagement * adjusted_sentiment * confidence
            final_score = standardize_and_normalize(normalized)  # Scale to [-1,1]
            """)

    def render_account_section(self):
        st.subheader("Account Analysis")
        
        with st.expander("Suspicious Activity Score (0-3)"):
            st.markdown("""
            Points accumulate based on:
            1. **High Posting Frequency**: +1 if posts/day > 95th percentile
            2. **Low Follower Count**: +1 if followers < 10th percentile
            3. **New Account**: +1 if account age < 30 days
            
            Score interpretation:
            - 0-1: Normal activity
            - 2: Potentially suspicious
            - 3: Highly suspicious
            """)
            
        with st.expander("Account Metrics"):
            st.markdown("""
            - **Account Age**: Days since account creation
            - **Posts per Day**: `total_user_posts / days_from_join_date`
            - **Follower Count**: Number of account followers
            """)

    def render_visualization_section(self):
        st.subheader("Visualization Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Choropleth Map")
            st.markdown("""
            - **State-level Sentiment**: Average RoBERTa-based sentiment scores by state
            - **Color Scale**: 
                - Red: Negative (-1)
                - White: Neutral (0)
                - Blue: Positive (1)
            """)
            
        with col2:
            st.markdown("#### User Influence Analysis")
            st.markdown("""
            - **X-axis**: Log-scaled follower count
            - **Y-axis**: Normalized sentiment score
            - **Bubble size**: Engagement score
            - **Color**: Hashtag category (Trump: Red, Biden: Blue, Both: Green)
            """)

    def render_data_processing_section(self):
        st.subheader("Data Processing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Tweet Cleaning")
            st.markdown("""
            - Removes URLs, emojis, and special characters
            - Normalizes @mentions to @user
            - Filters non-English tweets
            - Removes duplicates and retweets
            """)
            
        with col2:
            st.markdown("#### Text Processing")
            st.markdown("""
            - **Stopwords Removal**: Eliminates common words
            - **Lemmatization**: Reduces words to base form
            - **Tokenization**: Splits text into individual tokens
            - **Duplicate Handling**: 
                - Removes retweets
                - Labels cross-hashtag duplicates as 'both'
                - Keeps first occurrence of exact duplicates
            """)

if __name__ == "__main__":
    GlossaryPage()
