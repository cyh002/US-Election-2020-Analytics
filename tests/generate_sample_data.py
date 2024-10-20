import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta, time
from src.preprocessing import cast_data_type

def create_sample_data() -> pd.DataFrame:
    """
    Generates a sample DataFrame with 100 rows for prototyping a Streamlit app.
    
    Returns:
        pd.DataFrame: A DataFrame containing sample tweet data.
    """
    # Define all US states and District of Columbia
    states = [
        'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado',
        'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho',
        'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana',
        'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota',
        'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada',
        'New Hampshire', 'New Jersey', 'New Mexico', 'New York',
        'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon',
        'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota',
        'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington',
        'West Virginia', 'Wisconsin', 'Wyoming', 'District of Columbia'
    ]
    
    # Define possible sources
    sources = [
        'Twitter Web App', 'Twitter for iPhone', 'Twitter for Android',
        'Instagram', 'Facebook', 'Buffer', 'Hootsuite', 'TweetDeck'
    ]
    
    # Define possible hashtags
    hashtags = ['biden', 'trump', 'both']
    
    # Define possible sentiments
    sentiments = [-1, 0, 1]
    
    # Define user descriptions (sample)
    user_descriptions = [
        "Lifelong Democrat. Joe Biden for President 2024.",
        "Conservative thinker and writer.",
        "Activist and community organizer.",
        "Tech enthusiast and blogger.",
        "Student at XYZ University.",
        "Freelance journalist covering politics.",
        "Marketing professional and entrepreneur.",
        "Teacher and education advocate.",
        "Healthcare worker and volunteer.",
        "Artist and musician.",
        "Engineer with a passion for renewable energy.",
        "Lawyer specializing in civil rights.",
        "Small business owner.",
        "Environmental scientist.",
        "Retired military officer.",
        "Parent and local sports coach.",
        "Non-profit director.",
        "Financial analyst.",
        "Photographer and traveler.",
        "Stay-at-home parent."
    ]
    
    # Define sample tweet templates
    tweet_templates = [
        "Check out my latest post about #{}!",
        "I can't believe what's happening with #{}.",
        "Support #{} for a better future.",
        "The impact of #{} is undeniable.",
        "Why #{} matters more than ever.",
        "Join me in supporting #{}.",
        "Discussing the latest on #{} today.",
        "Here's my take on #{}.",
        "The truth about #{} revealed.",
        "Excited for what's next with #{}!"
    ]
    
    # Initialize lists to store data
    data = {
        'tweet_id': [],
        'tweet': [],
        'likes': [],
        'retweet_count': [],
        'source': [],
        'user_id': [],
        'user_id_post_count': [],
        'user_description': [],
        'days_from_join_date': [],
        'user_followers_count': [],
        'state': [],
        'hashtag': [],
        'clean_tweet': [],
        'no_stopwords': [],
        'sentiment': [],
        'confidence': [],
        'engagement': [],
        'normalized_scores': [],
        'created_date': [],
        'created_time': []
    }
    
    # Define start and end dates for created_date
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2024, 10, 19)
    delta_days = (end_date - start_date).days
    
    for i in range(1, 10000):
        # Generate tweet_id as a large unique number string
        tweet_id = str(random.randint(10**17, 10**18 - 1))
        
        # Select a random hashtag category
        hashtag_category = random.choice(hashtags)
        
        # Select a random tweet template and format it with the hashtag
        tweet = random.choice(tweet_templates).format(hashtag_category)
        
        # Generate likes and retweet_count
        likes = random.randint(0, 1000)
        retweet_count = random.randint(0, 500)
        
        # Select a random source
        source = random.choice(sources)
        
        # Generate user_id as a unique string
        user_id = str(random.randint(10**7, 10**9))
        
        # Generate user_id_post_count
        user_id_post_count = random.randint(1, 100)
        
        # Select a random user description
        user_description = random.choice(user_descriptions)
        
        # Generate days_from_join_date
        days_from_join_date = random.randint(0, 3650)  # Up to 10 years
        
        # Generate user_followers_count
        user_followers_count = random.randint(0, 100000)
        
        # Select a random state
        state = random.choice(states)
        
        # Assign hashtag based on category
        if hashtag_category == 'biden':
            hashtag = 'biden'
        elif hashtag_category == 'trump':
            hashtag = 'trump'
        else:
            hashtag = 'both'
        
        # Clean tweet (simple lowercase and remove special characters for example)
        clean_tweet = tweet.lower().replace('#', '').replace('@', '').replace('!', '').replace('.', '').replace(',', '')
        
        # Remove stopwords (for simplicity, removing common words)
        stopwords = {'the', 'and', 'a', 'to', 'for', 'of', 'in', 'on', 'with', 'is', 'it', 'this', 'that'}
        no_stopwords = [word for word in clean_tweet.split() if word not in stopwords]
        
        # Generate sentiment
        sentiment = random.choice(sentiments)
        
        # Generate confidence
        confidence = round(random.uniform(0.1, 1.0), 6)
        
        # Calculate engagement
        # To avoid division by zero, add 1 to followers
        engagement = round((likes + retweet_count) / (user_followers_count + 1) * 2, 6)
        engagement = min(max(engagement, 0.0), 2.0)  # Ensure engagement is between 0 and 2
        
        # Calculate normalized_scores
        normalized_scores = round(engagement * sentiment * confidence, 6)
        
        # Generate created_date
        random_days = random.randint(0, delta_days)
        created_date = (start_date + timedelta(days=random_days)).date()
        
        # Generate created_time
        random_seconds = random.randint(0, 86399)
        created_time = (datetime.min + timedelta(seconds=random_seconds)).time()
        
        # Append data to lists
        data['tweet_id'].append(tweet_id)
        data['tweet'].append(tweet)
        data['likes'].append(likes)
        data['retweet_count'].append(retweet_count)
        data['source'].append(source)
        data['user_id'].append(user_id)
        data['user_id_post_count'].append(user_id_post_count)
        data['user_description'].append(user_description)
        data['days_from_join_date'].append(days_from_join_date)
        data['user_followers_count'].append(user_followers_count)
        data['state'].append(state)
        data['hashtag'].append(hashtag)
        data['clean_tweet'].append(clean_tweet)
        data['no_stopwords'].append(no_stopwords)
        data['sentiment'].append(sentiment)
        data['confidence'].append(confidence)
        data['engagement'].append(engagement)
        data['normalized_scores'].append(normalized_scores)
        data['created_date'].append(created_date)
        data['created_time'].append(created_time)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Optionally, shuffle the DataFrame
    df = df.sample(frac=1).reset_index(drop=True)
    
    return df

# Example usage:
if __name__ == "__main__":
    sample_df = create_sample_data()
    sample_df = cast_data_type(sample_df)
    # save the sample data to a CSV file
    sample_df.to_csv('sample_data.csv', index=False)
    print(sample_df.head())
