#!/usr/bin/env python
# coding: utf-8

# In[1]:


import flask
import re
import random
import string
from datetime import datetime
from progressbar import ProgressBar
import pandas as pd
pbar = ProgressBar()
import collections
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import glob
import seaborn as sns
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from gensim import corpora, models, similarities, matutils
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
import numpy as np
import pickle
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en.stop_words import STOP_WORDS
from langdetect import detect 
from textblob import TextBlob
from langdetect import detect_langs

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import pairwise_distances


# In[2]:


app = flask.Flask(__name__)  # create instance of Flask class

with open("doc_topic_nmf.pkl", "rb") as f:
    doc_topic_nmf = pickle.load(f)
    
with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)
    
with open("nmf_model.pkl", "rb") as f:
    nmf_model = pickle.load(f)
    
with open("reviews.pkl", "rb") as f:
    reviews = pickle.load(f)
    
with open("doc_word.pkl", "rb") as f:
    doc_word = pickle.load(f)


# In[3]:


# minimal example from:
# http://flask.pocoo.org/docs/quickstart/
    
def lang(sentence):
    try:
        return detect(sentence)
    except:
        None

def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def clean_text(text):
    # lower text
    text = text.lower()
    # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove stop words
    stop = stopwords.words('english')
    manual_stop = ['bumble', 'tinder', 'zoosks', 'zoosk', 'bumble', 'east meets east', 'grindr', 'hinge',
                  'cmb', 'okcupid','bumble', 'tinders', 'zoosks', 'zoosk', 'bumbles', 'east meets east', 'grindrs', 'hinges',
                  'cmbs', 'okcupids']
    stop.extend(manual_stop)
    text = [x for x in text if x not in stop]
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    # pos tag text
    pos_tags = pos_tag(text)
    # lemmatize text
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    # join all
    text = " ".join(text)
    return(text)

def clean_again_text(text):
    # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # remove stop words
    stop = stopwords.words('english')
    manual_stop = ['bumble', 'tinder', 'zoosks', 'zoosk', 'bumble', 'east meets east', 'grindr', 'hinge','okc',
                  'cmb', 'okcupid','bumble', 'tinders', 'zoosks', 'zoosk', 'bumbles', 'east meets east', 'grindrs', 'hinges',
                  'cmbs', 'okcupids', 'im', 'i','think','ive', 'want', 'lol', 'haha' , 'cupid', 'ok', 'let', 'know',
                  'right','dont','lot','happn','league','blk','jswipe']
    stop.extend(manual_stop)
    stop.extend(STOP_WORDS)
    
    text = [x for x in text if x not in stop]
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    # join all
    text = " ".join(text)
    return(text)
    
def get_related_app(phrase):
    phrase = clean_text(phrase)
    phrase = re.sub('([^\x00-\x7F])+', '', phrase)
    phrase = re.sub(r"[\/\-\'\)\(,.;@#?!&$]+\ *", " ", phrase)
    phrase = clean_again_text(phrase)
    phrase = [phrase]
    phrase_doc_word = tfidf_vectorizer.transform(phrase)
    phrase_doc_topic = nmf_model.transform(phrase_doc_word)
    df_doc = doc_topic_nmf
    d = pairwise_distances(doc_word,phrase_doc_word,metric='cosine')
    #[films[m] for m in d[1].argsort()[:10]]
    df = pd.DataFrame(data=d, columns=["signal"])
    df_merged = df.merge(df_doc, how='inner', left_index=True, right_index=True, left_on=None, right_on=None)
    df_fin_merged = df_merged.merge(reviews, how='inner', left_index=True, right_index=True)
    _ = df_fin_merged[df_fin_merged['stars'] >= 4].sort_values('signal', ascending=True).head(1)
    return(_['app'])
    


# In[ ]:



@app.route('/')  # the site to route to, index/main in this case
def luanch():
    return flask.render_template('predict_app.html')

@app.route("/predict"#, methods=["POST","GET"]
          )
def predict():
    print(flask.request.args)
    term = flask.request.args
    app = get_related_app(term.get('term'))
    print(app)
    return flask.render_template('predict_app.html', prediction=app)

if __name__ == '__main__':
    app.run(host='127.0.0.1',port=5001)


# In[ ]:





# In[ ]:




