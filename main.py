import nltk
import numpy as np
import random
import sklearn
import time
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
import warnings
from nltk.lm.preprocessing import pad_both_ends

warnings.filterwarnings("ignore")
# Read Corpus
# Converting every text to lower case
df = pd.read_csv('Dataset.csv')
qa_in = list(df['Question'])
qa_in = ' '.join(str(e) for e in qa_in).lower()
qa_ans = list(df['Answer'])
qa_ans = ' '.join(str(e) for e in qa_ans).lower()

df = pd.read_csv('smalltalk.csv')
smalltalk_in = list(df['Question'])
smalltalk_in = ' '.join(str(e) for e in smalltalk_in).lower()
smalltalk_ans = list(df['Answer'])
smalltalk_ans = ' '.join(str(e) for e in smalltalk_ans).lower()

# Using Punkt tokenizer
# nltk.download('punkt')
# Using wordnet dictionary
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('universal_tagset')
# nltk.download('vader_lexicon')

# Corpus Tokenization (Split a string into meaningful units)
qa_in_token = nltk.sent_tokenize(qa_in, language='english')
qa_ans_token = nltk.sent_tokenize(qa_ans, language='english')
print(qa_in_token)
smalltalk_in_token = sent_tokenize(smalltalk_in, language='english')
smalltalk_ans_token = sent_tokenize(smalltalk_ans, language='english')
print(smalltalk_in_token)
# sentence_tokens = nltk.sent_tokenize(df)
# word_tokens = nltk.word_tokenize(df)

lemmer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def lemToke(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


remove_punc = dict((ord(punct), None) for punct in string.punctuation)

# string.punctuation = string.punctuation + '“' + '”' + '-' + '’' + '‘' + '—'
# string.punctuation = string.punctuation.replace('.', '')
def lemNormalize(text):
    return lemToke(word_tokenize(text.translate(remove_punc)))


def random_response():
    random_list = [
        "Please try writing something more descriptive.",
        "Oh! It appears you wrote something I don't understand yet!",
        "Do you mind trying to rephrase that?",
        "I'm terribly sorry, I didn't quite catch that.",
        "I can't answer that yet, please try asking something else."
    ]
    count = len(random_list)
    random_item = random.randrange(count)
    return random_list[random_item]


# Similarity Function
def similarity(token, user_res_len):
    if (user_res_len <= 5):
        TfidfVec = TfidfVectorizer(tokenizer=lemNormalize, min_df=0.01)
    else:
        TfidfVec = TfidfVectorizer(tokenizer=lemNormalize, min_df=0.01, stop_words='english')
    tfidf = TfidfVec.fit_transform(token)
    vals = cosine_similarity(tfidf[-1], tfidf)
    return vals


def qa_response(user_response, token):
    bot_response = ''
    token.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=lemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(token)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    token.remove(user_response)
    if req_tfidf == 0:
        bot_response = bot_response + random_response()
        return bot_response
    else:
        bot_response = bot_response + token[idx]
        return bot_response


def st_response(token, token2):
    bot_response = ''
    TfidfVec = TfidfVectorizer(tokenizer=lemNormalize)
    tfidf = TfidfVec.fit_transform(token)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        bot_response = bot_response + random_response()
        return bot_response
    else:
        bot_response = bot_response + token2[idx]
        return bot_response


greeting_in = ['whats up', 'how are you', 'how are you doing', "how its going", 'what are you feeling today', 'hello', 'hi']
greeting_out = ['hi😎', 'hey😎', 'hey there!😎', 'nods😎', "what's up😎, I am good", "what's good brother😎"]

# Time Questions Dataset
time_in = ['good morning', 'good evening', 'good afternoon', 'good night', 'can you tell me the date today',
           'what day is today', 'what is the time now']
name_ques = ['what is my name', 'do you know my name', 'who am i', 'do you know me', 'do you know who am i',
             'do you remember my name']


# Intent Routing
def intent_route(user_response):
    user_res_len = len(lemNormalize(user_response))

    qa_in_token.append(user_response)
    smalltalk_in_token.append(user_response)
    greeting_in.append(user_response)
    time_in.append(user_response)
    name_ques.append(user_response)

    ans_val = np.mean(similarity(qa_in_token, user_res_len))
    smalltalk_val = np.mean(similarity(smalltalk_in_token, user_res_len))
    greeting_val = np.mean(similarity(greeting_in, user_res_len))
    time_val = np.mean(similarity(time_in, user_res_len))
    name_val = np.mean(similarity(name_ques, user_res_len))

    qa_in_token.remove(user_response)

    val_arr = [ans_val, smalltalk_val, greeting_val, time_val, name_val]
    print(val_arr)
    if max(val_arr) < 0.2:
        if smalltalk_val < ans_val:
            return qa_response(user_response, qa_ans_token)
        else:
            return st_response(smalltalk_in_token, smalltalk_ans_token)
    else:
        idx = np.argsort(val_arr, None)[-1]
        if idx == 0:
            return qa_response(user_response, qa_ans_token)

        elif idx == 1:
            return st_response(smalltalk_in_token, smalltalk_ans_token)

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
            return 'QQ Bot: Your name is ' + name + ', would not forget who my master is!😎'


# def greeting(sentence):
#     for word in sentence.split():
#         if word.lower() in greeting_in:
#             return random.choice(greeting_out)


# def response(user_response, token2, tfidf=None):
#     robo1_response = ''
#     # Perform tokenization using LemNormalize and handle stopwords by a default of english
#     TfidVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
#     # Fit a transform and convert everything into a tfidf vector
#     tfidf - TfidVec.fit_transform(user_response)
#     # Do cosine similarity after that
#     vals = cosine_similarity(tfidf[-1], tfidf)
#     # Do sort and return the most similar or the first element within the set of values
#     idx = vals.argsort()[0][-2]
#     flat = vals.flatten()
#     flat.sort()
#     req_tfidf = flat[-2]
#     if req_tfidf == 0:
#         robo1_response = robo1_response + "I am sorry. Unable to understand you!😵"
#         return robo1_response
#     else:
#         robo1_response = robo1_response + token2[idx]
#         return robo1_response

# Identity Management
print("QQ Bot: Hi, I'm QQ Bot. May I know what's your name?")
print('')
empty = True
while empty == True:
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
while flag == True:
    user_response = input(name + ': ')
    user_response = user_response.lower().translate(remove_punc)

    if user_response != 'bye':

        if user_response == 'thank you' or user_response == 'thanks':
            print('')
            print('QQ Bot: You are Welcome..😎 I am always here to help!')
            print('')

        else:
            print('')
            print('QQ Bot: ', end="")
            print(intent_route(user_response))
            smalltalk_in_token.remove(user_response)
            greeting_in.remove(user_response)
            time_in.remove(user_response)
            print('')
            print('QQ Bot: Do you have any other questions for me to answer you?')
            print('')
    else:
        flag = False
        print('QQ Bot: Goodbye ' + name + '👋! See you soon!')

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
