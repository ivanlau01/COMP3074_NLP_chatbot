import numpy as np
import random
import sklearn
import pandas as pd

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from nltk.tokenize import word_tokenize
from nltk import sent_tokenize


import string
from nltk.lm.preprocessing import pad_both_ends

data_path = 'smalltalk.csv'
N_PARAM = 2  # max ngram length of the language model

text_string = text.lower()

# Remove punctuation
string.punctuation = string.punctuation + '“' + '”' + '-' + '’' + '‘' + '—'
string.punctuation = string.punctuation.replace('.', '')  # keep "." so that can split sentences with NLTK
text_filtered = "".join([char for char in text_string if char not in string.punctuation])
text_sentences = sent_tokenize(text_filtered)

text_tokenized = [word_tokenize(sentence) for sentence in text_sentences]
text_padded = [list(pad_both_ends(sentence_tokenized, n=N_PARAM))
               for sentence_tokenized in text_tokenized]