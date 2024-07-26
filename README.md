import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Download the VADER lexicon
nltk.download('vader_lexicon')

# Load the dataset
file = '/content/drive/MyDrive/dataset2.csv'
dt = pd.read_csv(file)
# Check the column names
print(dt.columns)

# Initialize the SentimentIntensityAnalyzer
side = SentimentIntensityAnalyzer()

# Define a function to get sentiment scores
def get_sentiment_scores(text):
    return side.polarity_scores(text)

# Replace 'name' with the correct text column name

text_column = 'name'

# change this to your actual text column name

# Check if the text column exists in the dataset

if text_column in dt.columns:

    dt['sentiment_scores'] = dt[text_column].apply(get_sentiment_scores)
    dt['compound_score'] = dt['sentiment_scores'].apply(lambda x: x['compound'])
    dt['sentiment'] = dt['compound_score'].apply(lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral'))



    print(dt[[text_column, 'sentiment_scores', 'compound_score', 'sentiment']].head())


    plt.figure(figsize=(10, 6))
    sns.histplot(dt['compound_score'], kde=True)
    plt.title('Distribution of compound Scores')
    plt.xlabel('Compound Score')
    plt.ylabel('Frequency')
    plt.show()

    # Plot count of sentiment classifications
    plt.figure(figsize=(10, 6))
    sns.countplot(x='sentiment', data=dt, palette='viridis')
    plt.title('Count of Sentiment Classifications')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.show()

    # Analyze sentiment by genre
    if 'genre' in dt.columns:
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='genre', y='compound_score', data=dt)
        plt.title('Sentiment by Genre')
        plt.xlabel('Genre')
        plt.ylabel('Compound Score')
        plt.xticks(rotation=90)
        plt.show()

    # Analyze sentiment by rating
    if 'rating' in dt.columns:
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='rating', y='compound_score', data=dt)
        plt.title('Sentiment by Rating')
        plt.xlabel('Rating')
        plt.ylabel('Compound Score')
        plt.xticks(rotation=90)
        plt.show()

    # Correlation matrix of numerical features
    numerical_features = ['favorability', 'year', 'votes', 'budget', 'gross', 'runtime', 'compound_score']
    corr_matrix = dt[numerical_features].corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix of Numerical Features')
    plt.show()

else:
    print(f"Column '{text_column}' not found in the dataset.")
