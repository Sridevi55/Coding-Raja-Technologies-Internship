import streamlit as st
import re
from cleantext import clean
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer


def clean_text(text):
    text = re.sub(r"@[\S]+", "", text)
    return clean(
        text,
        to_ascii=True,
        fix_unicode=True,
        no_urls=True,
        replace_with_url="",
        no_currency_symbols=True,
        replace_with_currency_symbol="",
        no_punct=True,
        no_digits=True,
        replace_with_digit="",
    )


stop_words = [
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', "i", 'for', 'from', 'has',
    'have', 'he', 'her', 'him', 'his', 'in', 'is', 'it', 'its', 'of', 'on',
    'or', 'that', 'the', 'to', 'was', 'were', 'with', 'should', 'have', 'has', 'would', 'should', 'could', 'would', 'might',
    'must', 'shall', 'ought', 'need', 'dare', 'because', 'while', 'when', 'where', 'after',
    'before', 'although', 'though', 'if', 'unless', 'until', 'while', 'even', 'once', 'since',
    'so', 'than', 'that', 'though', 'till', 'when', 'whenever', 'whereas', 'wherever', 'whether',
    'which', 'while', 'who', 'whoever', 'whom', 'whose', 'how', 'why', 'what', 'where', 'when',
    'which', 'who', 'whom', 'this', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and',
    'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
    'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', "im", "wa", "ive"
]


def remove_stopwords_stem(text):
    nltk.download("wordnet")
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(t) for t in text.split(
    ) if lemmatizer.lemmatize(t) not in stop_words]
    return " ".join(text)


st.title("Sentiment Analyzer 😀😭😐")
text = st.text_area("Enter text")

cleaned_text = clean_text(text)
with st.spinner("Processing..."):
    stemmed_text = remove_stopwords_stem(text).strip()


@st.cache_resource
def load_my_model():
    try:
        model = tf.keras.models.load_model(
            filepath="./saved_models/lstm_model/model4_lstm_new.pb")
    except Exception as e:
        st.write(e)
    return model


model = load_my_model()
sentiments = ["Negative", "Neutral", "Positive"]


if st.button("Predict"):
    y_pred = model.predict([stemmed_text])
    y_pred_ind = np.squeeze(np.argmax(y_pred, axis=1))
    if y_pred_ind == 0:
        st.markdown(f"### Predicted sentiment: :red[{sentiments[y_pred_ind]}]")
    elif y_pred_ind == 1:
        st.markdown(
            f"### Predicted sentiment: :blue[{sentiments[y_pred_ind]}]")
    else:
        st.markdown(
            f"### Predicted sentiment: :green[{sentiments[y_pred_ind]}]")
