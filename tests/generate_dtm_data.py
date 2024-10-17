# to generate test_dtm.csv data
import pandas as pd

# Define the data
data = {
    'created_date': ['2020-10-15', '2020-10-15'],  # Dates as strings (can be converted later)
    'created_time': ['00:00:02', '00:00:08'],  # Times as strings
    'tweet_id': [132.0, 133.0],  # Float values
    'tweet': ['#Trump: As a student...', 'You get a tie!'],  # String values
    'likes': ['2', '4'],  # String values (could be integers after conversion)
    'retweet_count': [10.0, 20.0],  # Float values
    'source': ['Twitter Web App', 'Twitter for iPhone'],  # String values
    'user_id': ['8436472', '47413798'],  # String values
    'user_name': ['snarke', 'Rana Abtar'],  # String values
    'user_screen_name': ['snarke', 'Ranaabtar'],  # String values
    'user_description': ['Freelance writer...', 'Washington Correspondent...'],  # String values
    'days_from_join_date': [7, 30],  # Integer values
    'user_join_date': ['2007-08-26 05:56:11', '2009-06-15 19:05:35'],  # Datetime as strings
    'user_followers_count': [200, 300],  # Integer values
    'user_location': ['Portland', 'Washington DC'],  # String values
    'lat': [-122.6741949, -77.0365581],  # Float values
    'long': ['-122.6741949', '-77.0365581'],  # String values (could convert to float if consistent)
    'country': ['United States of America', 'United States of America'],  # String values
    'continent': ['North America', 'North America'],  # String values
    'state': ['Oregon', 'District of Columbia'],  # String values
    'state_code': ['OR', 'DC'],  # String values
    'days_from_collection': [7, 10],  # Integer values
    'hashtag': ['trump', 'biden'],  # String values
    'clean_tweet': ['#Trump: As a student...', 'You get a tie!'],  # String values
    'no_stopwords': [['#trump', 'student', 'used'], ['#trump', 'rally', '#iowa']]  # List of strings
}

# Convert the data into a DataFrame
df = pd.DataFrame(data)

# Cast columns to appropriate data types where necessary
df['created_date'] = pd.to_datetime(df['created_date'], format='%Y-%m-%d')
df['created_time'] = pd.to_datetime(df['created_time'], format='%H:%M:%S').dt.time
df['likes'] = pd.to_numeric(df['likes'], errors='coerce')
df['user_id'] = df['user_id'].astype(str)
df['long'] = pd.to_numeric(df['long'], errors='coerce')

# Save the DataFrame to a CSV file
df.to_csv('test_dtm.csv', index=False)
