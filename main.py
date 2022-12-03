import nltk
import numpy as np
import random
import sklearn
import pandas as pd

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from nltk.tokenize import word_tokenize
from datetime import datetime
from nltk import sent_tokenize
from nltk.stem import WordNetLemmatizer

import string
from nltk.lm.preprocessing import pad_both_ends

df = pd.read_csv('smalltalk.csv')
# Converting every text to lower case
smalltalk_ques = list(df['Question'])
smalltalk_ques = ' '.join(str(e) for e in smalltalk_ques).lower()
smalltalk_ans = list(df['Answer'])
smalltalk_ans = ' '.join(str(e) for e in smalltalk_ans).lower()
# Using Punkt tokenizer
# nltk.download('punkt')
# Using wordnet dictionary
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# Corpus Tokenization (Split a string into meaningful units)
smalltalk_ques_token = word_tokenize(smalltalk_ques, language='english')
smalltalk_ans_token = word_tokenize(smalltalk_ans, language='english')
print(smalltalk_ans_token)
# sentence_tokens = nltk.sent_tokenize(df)
# word_tokens = nltk.word_tokenize(df)

lemmer = WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


remove_punc_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(word_tokenize(text.translate(remove_punc_dict)))
print()
greet_inputs = ('hello', 'hi', 'wassuppp', 'how are you?')
greet_responses = ('hi', 'hey', 'hey there!', 'there there!!')


def greet(sentence):
    for word in sentence.split():
        if word.lower() in greet_inputs:
            return random.choice(greet_responses)


def response(user_response, token2, tfidf=None):
    robo1_response = ''
    # Perform tokenization using LemNormalize and handle stopwords by a default of english
    TfidVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    # Fit a transform and convert everything into a tfidf vector
    tfidf - TfidVec.fit_transform(user_response)
    # Do cosine similarity after that
    vals = cosine_similarity(tfidf[-1], tfidf)
    # Do sort and return the most similar or the first element within the set of values
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        robo1_response = robo1_response + "I am sorry. Unable to understand you!"
        return robo1_response
    else:
        robo1_response = robo1_response + token2[idx]
        return robo1_response

flag = True
print('Hello! I am the Learning Bot. Start typing your text after greeting talk to me. For ending the '
          'conversation type bye!')
while (flag == True):
    user_response = input()
    user_response = user_response.lower()
    if (user_response != 'bye'):
        if (user_response == 'thank you' or user_response == 'thanks'):
            flag = False
            print('DoomBot: You are Welcome..')
        else:
            if (greet(user_response) != None):
                print('DoomBot: ' + greet(user_response))
            else:
                smalltalk_ans_token.append(user_response)
                word_tokens = word_tokens + nltk.word_tokenize(user_response)
                finals_words = list(set(word_tokens))
                print('DoomBot:', end='')
                print(response(user_response))
                smalltalk_ans_token.remove(user_response)
    else:
        flag = False
        print('DoomBot: Goodbye!')

# N_PARAM = 2  # max ngram length of the language model
# # Remove punctuation
# string.punctuation = string.punctuation + '“' + '”' + '-' + '’' + '‘' + '—'
# string.punctuation = string.punctuation.replace('.', '')  # keep "." so that can split sentences with NLTK
# text_filtered = "".join([char for char in raw_data if char not in string.punctuation])
# text_sentences = sent_tokenize(text_filtered)
# 
# text_tokenized = [word_tokenize(sentence) for sentence in text_sentences]
# text_padded = [list(pad_both_ends(sentence_tokenized, n=N_PARAM))
#                for sentence_tokenized in text_tokenized]