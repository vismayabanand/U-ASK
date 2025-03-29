import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('C:\\Users\\ranjitha\\OneDrive\\Desktop\\U-ASK\\processed_tweets.csv')

# Remove duplicates based on 'tweet_id' column
df = df.drop_duplicates(subset=['tweet_id'])

# Save the cleaned CSV to a new file
df.to_csv('C:\\Users\\ranjitha\\OneDrive\\Desktop\\U-ASK\\tweets_cleaned.csv', index=False)
