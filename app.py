import re
import gradio as gr
import pandas as pd
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk


# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet

lemmatizer= WordNetLemmatizer()

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts."""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {
        'J': wordnet.ADJ,
        'N': wordnet.NOUN,
        'V': wordnet.VERB,
        'R': wordnet.ADV
    }
    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize_content(content):
    lemmatized_content = re.sub('[^a-zA-Z]', ' ', content)
    lemmatized_content = lemmatized_content.lower()
    lemmatized_content = lemmatized_content.split()
    lemmatized_content = [
        lemmatizer.lemmatize(word, get_wordnet_pos(word))
        for word in lemmatized_content
        if word not in stopwords.words('english')
    ]
    lemmatized_content = ' '.join(lemmatized_content)
    return lemmatized_content

#load the model
model = tf.keras.models.load_model('twitter_sentiment_analysis_epoch4.h5')

loaded_model = tf.keras.models.load_model('vectorizer_model')
loaded_vectorizer = loaded_model.layers[0]

def score_comment(comment):
    # Preprocess the comment
    comment = lemmatize_content(comment)
    # Vectorize the input comment
    vectorized_comment = loaded_vectorizer([comment])
    # Predict using the loaded model
    result = model.predict(vectorized_comment)

    # Generate the output text based on predictions
    text = ''
    if result<0.5:
       text= 'Negative'
    else:
       text = 'Positive'
    return text

interface = gr.Interface(
    fn=score_comment,
    inputs=gr.Textbox(lines=2, placeholder='Tweet to score'),
    outputs='text'
)

interface.launch()