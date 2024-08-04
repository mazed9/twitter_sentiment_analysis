# Twitter Sentiment Analysis

This project implements a sentiment analysis model to predict the sentiment (positive or negative) of tweets. An LSTM-based model has been trained on 1.6 million tweets.

## Project Structure

- __01. Data Preparation:__
  * `Data Collection`: The dataset consisting 1.6 million tweets has been collected from [here](https://www.kaggle.com/datasets/kazanova/sentiment140).
  * `Data Cleaning & Preprocessing`:
    - Removed stopwords
    - Applied Lemmatization
    - Vectorized the lemmatized data utilizing "TextVectorization" from keras
    - Saved the Vectorizer for utilizing later in the app
  
- __02. Model Training:__
  * A Bidirectional LSTM model with an embedding layer has been trained on the preprocessed data.
  
- __03. App Deployment:__
  * Developed a web-app with Gradio interface
  * Deployed the [App](https://huggingface.co/spaces/mazed/twitter_sentiment_analysis) in HuggingFace Spaces

- `requirements.txt`: Contains the dependencies needed for the project:
  - `pandas`
  - `tensorflow==2.15.0`
  - `nltk`
  - `gradio`
