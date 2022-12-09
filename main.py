import numpy as np
import random
import pandas as pd
import nltk

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from datetime import datetime
from nltk.stem import WordNetLemmatizer

import string
import warnings

warnings.filterwarnings("ignore")

# Read Corpus
# Converting every text to lower case
qa_df = pd.read_csv('QnADataset.csv')
qa_in = list(qa_df['Question'])

st_df = pd.read_csv('smalltalk.csv')
smalltalk_in = list(st_df['Question'])

# Run it on the first run
# Using Punkt tokenizer
# nltk.download('punkt')

# Using wordnet dictionary
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('stopwords')

lem = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


# Pre-processing
def tokenization(tokens):
    return [lem.lemmatize(token) for token in tokens]


remove_punc = dict((ord(punc), None) for punc in string.punctuation)


def lemmatization(text):
    return tokenization(word_tokenize(text.translate(remove_punc).lower()))


# Response for not understanding the requirement from users
def random_response():
    random_list = [
        "😵 Please try writing something more descriptive.",
        "Oh! It appears you wrote something I don't understand yet! Could you rephrase it for me again please?",
        "😵 Do you mind trying to rephrase that again please?",
        "😵 I'm terribly sorry, I didn't quite catch that. Could you rephrase it for me again please?",
        "😵 I can't answer that yet, please try asking something else."
    ]
    count = len(random_list)
    random_item = random.randrange(count)
    return random_list[random_item]


# Similarity Function
def similarity(token, query):
    TfidfVec = TfidfVectorizer(tokenizer=lemmatization, min_df=0.01)
    tfidf = TfidfVec.fit_transform(token).toarray()
    tfidf_query = TfidfVec.transform([query]).toarray()
    vals = cosine_similarity(tfidf_query, tfidf)
    return vals


# Question and answer response function
def qa_response(user_in, token):
    TfidfVec = TfidfVectorizer(tokenizer=lemmatization, stop_words='english')
    tfidf = TfidfVec.fit_transform(token)
    tfidf_query = TfidfVec.transform([user_in]).toarray()
    vals = cosine_similarity(tfidf_query, tfidf)
    idx = np.argmax(vals)

    if vals.max() > 0:
        return qa_df['Answer'][idx]
    else:
        return random_response()


# Small talk response function
def st_response(user_in, token2):
    TfidfVec = TfidfVectorizer(tokenizer=lemmatization)
    tfidf = TfidfVec.fit_transform(token2)
    tfidf_query = TfidfVec.transform([user_in]).toarray()
    vals = cosine_similarity(tfidf_query, tfidf)
    idx = np.argmax(vals)

    if vals.max() > 0:
        return st_df['Answer'][idx]
    else:
        return random_response()


# Greetings dataset
greeting_in = ['whats up', 'how are you', 'how are you doing', "how's it going", 'what are you feeling today', 'hello',
               'hi', 'hey', 'greetings']
greeting_out = ['Hi😎', 'Hey😎', 'Hey there!😎', '*nods*😎', "What's up😎", "What's good brother😎"]

# Time Questions Dataset
time_in = ['good morning', 'good evening', 'good afternoon', 'good night', 'can you tell me the date today',
           'what day is today', 'what is the date today', 'what is the time now', 'time', 'today']
# Name Questions Dataset
name_ques = ['what is my name', 'do you know my name', 'who am i', 'do you know me', 'do you know who am i',
             'do you remember my name', 'do you recognise me']


# Intent Matching
def intent_route(user_res):
    ans_val = similarity(qa_in, user_res).max()
    smalltalk_val = similarity(smalltalk_in, user_res).max()
    greeting_val = similarity(greeting_in, user_res).max()
    time_val = similarity(time_in, user_res).max()
    name_val = similarity(name_ques, user_res).max()

    val_arr = [ans_val, smalltalk_val, greeting_val, time_val, name_val]

    if max(val_arr) < 0.5:
        return qa_response(user_res, qa_in)
    else:
        idx = np.argsort(val_arr, None)[-1]
        if idx == 0:
            return qa_response(user_res, qa_in)

        elif idx == 1:
            return st_response(user_res, smalltalk_in)

        elif idx == 2:
            list_count = len(greeting_out)
            random_arrange = random.randrange(list_count)
            return greeting_out[random_arrange]

        elif idx == 3:
            now = datetime.now()
            currentTAD = now.strftime('%a %Y-%m-%d, %H:%M')
            currentTime = int(datetime.now().hour)
            if 5 <= currentTime < 12:
                return "Good Morning, it is " + currentTAD + " now! ⏰"
            elif 12 <= currentTime < 17:
                return "Good Afternoon, it is " + currentTAD + " now! ⏰"
            else:
                return "Good Evening, it is " + currentTAD + " now! ⏰"

        elif idx == 4:
            return 'Your name is ' + name + ', definitely would not forget who I am chatting with!😎'


# Identity Management
print("QQ Bot: Hi, I'm QQ Bot. May I know what's your name?😃")
print('')
empty = True
while empty:
    name = input()
    if name == "":
        print("QQ Bot: I'm sorry, can you re-enter your name?")
        print('')
    else:
        empty = False

# Chatbot interface
flag = True
print('')
print('QQ Bot: Hi ' + name + '🤝! I am QQ Bot, you may ask me any questions now✌. If you want to end the '
                             'conversation please type bye!')
print('')
while flag:
    user_response = input(name + ': ')
    user_response = user_response.lower().translate(remove_punc)

    if user_response != 'bye':

        if user_response == 'thank you' or user_response == 'thanks' or user_response == 'thank you so much':
            print('')
            print('QQ Bot: You are Welcome..😎 I am always here to help!')
            print('')

        else:
            print('')
            print('QQ Bot: ', end="")
            print(intent_route(user_response))
            print('')
    else:
        flag = False
        print('')
        print('QQ Bot: Goodbye ' + name + '👋! See you soon!')
