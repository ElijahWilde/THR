# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 18:23:19 2019

@author: Elijah Wilde with a lot of help from TowardsDataScience.com and StackOverflow.com
"""

# Import Libraries
import matplotlib.pyplot as plt
import json
import gzip
import re
import logging
import pandas as pd
import numpy as np
from numpy import random
import gensim
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
nltk.download('stopwords')
%matplotlib inline


# Importing the Data ==========================================================
books = json.load(gzip.open(r'C:\Users\Eli\Desktop\Python\THR\books.json.gz', "rt", encoding="utf-8"))
df = pd.DataFrame(books)
pd.set_option('display.max_colwidth', -1)


# Putting the Data in a Format That is Easy to Work With ======================
bookText=[]
bookGenre=[]
for row in df.itertuples():
    bookText.append(row.pages)
    if not row.categories:
        bookGenre.append("NoGen")
    else:
        bookGenre.append(row.categories[0])
        
tempString = ""
tempList = []
for x in bookText:
    for y in x:
        tempString = tempString + (y['text']) + " "  
    tempList.append(tempString)
    tempString = "" 

df = pd.DataFrame(
    {'Text': tempList,
     'Genre': bookGenre
    })  

df = df[df.Genre != 'NoGen']


# Visualizing the Data ========================================================
print(df.head(10))
print(df['Text'].apply(lambda x: len(x.split(' '))).sum())

my_tags = ['Alph', 'Anim', 'ArtM', 'Biog', 'Fair', 'Fict', 'Food', 'Heal', 'Hist', 'Holi', 'Math', 'Nurs', 'Peop', 'Poet', 'Recr', 'Spor']
plt.figure(figsize=(10,4))
df.Genre.value_counts().plot(kind='bar');

def print_plot(index):
    example = df[df.index == index][['Text', 'Genre']].values[0]
    if len(example) > 0:
        print(example[0])
        print('Genre:', example[1])
print_plot(10)


# Cleaning the Data ===========================================================
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    return text
    
df['Text'] = df['Text'].apply(clean_text)
print_plot(10)

X = df.Text
y = df.Genre
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)


# Naive Bayes Method ==========================================================
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
nb.fit(X_train, y_train)

from sklearn.metrics import classification_report
y_pred = nb.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=my_tags))


# Linear Support Vector Machine ===============================================
from sklearn.linear_model import SGDClassifier

sgd = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
               ])
sgd.fit(X_train, y_train)

y_pred = sgd.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=my_tags))


# Logistic Regression =========================================================
from sklearn.linear_model import LogisticRegression

logreg = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(n_jobs=1, C=1e5)),
               ])
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=my_tags))


# BOW With Keras ==============================================================
import itertools
import os

%matplotlib inline
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text, sequence
from keras import utils

train_size = int(len(df) * .7)
train_Text = df['Text'][:train_size]
train_Genre = df['Genre'][:train_size]

test_Text = df['Text'][train_size:]
test_Genre = df['Genre'][train_size:]

max_words = 1000
tokenize = text.Tokenizer(num_words=max_words, char_level=False)
tokenize.fit_on_texts(train_Text) # only fit on train

x_train = tokenize.texts_to_matrix(train_Text)
x_test = tokenize.texts_to_matrix(test_Text)

encoder = LabelEncoder()
encoder.fit(train_Genre)
y_train = encoder.transform(train_Genre)
y_test = encoder.transform(test_Genre)

num_classes = np.max(y_train) + 1
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)

batch_size = 32
epochs = 2

# Build the model
model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
              
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)

score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
print('Test accuracy:', score[1])


# Here is code adapted from an example on TowardsDataScience.com that should run Word2Vec and Doc2Vec
# I have modified it to work on my data but have not been able to test it as Gensim won't place nice on my OS.
# Doc2vec and Logistic Regression =============================================
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from sklearn import utils
import gensim
from gensim.models.doc2vec import TaggedDocument
import re

def label_sentences(corpus, label_type):
    labeled = []
    for i, v in enumerate(corpus):
        label = label_type + '_' + str(i)
        labeled.append(doc2vec.TaggedDocument(v.split(), [label]))
    return labeled
X_train, X_test, y_train, y_test = train_test_split(df.Text, df.Genre, random_state=0, test_size=0.3)
X_train = label_sentences(X_train, 'Train')
X_test = label_sentences(X_test, 'Test')
all_data = X_train + X_test


# Word2vec and Logistic Regression ============================================
from gensim.models import Word2Vec
wv = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)
# This word2vec model comes from google and has been already trained a 100 billion word Google News corpus.
wv.init_sims(replace=True) 

from itertools import islice
list(islice(wv.vocab, 13030, 13050))

def word_averaging(wv, words):
    all_words, mean = set(), []
    
    for word in words:
        if isinstance(word, np.ndarray):
            mean.append(word)
        elif word in wv.vocab:
            mean.append(wv.syn0norm[wv.vocab[word].index])
            all_words.add(wv.vocab[word].index)

    if not mean:
        logging.warning("cannot compute similarity with no input %s", words)
        # FIXME: remove these examples in pre-processing
        return np.zeros(wv.vector_size,)

    mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
    return mean

def  word_averaging_list(wv, text_list):
    return np.vstack([word_averaging(wv, Text) for Text in text_list ])

# Tokenize the text
def w2v_tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text, language='english'):
        for word in nltk.word_tokenize(sent, language='english'):
            if len(word) < 2:
                continue
            tokens.append(word)
    return tokens
    
train, test = train_test_split(df, test_size=0.3, random_state = 42)

test_tokenized = test.apply(lambda r: w2v_tokenize_text(r['Text']), axis=1).values
train_tokenized = train.apply(lambda r: w2v_tokenize_text(r['Text']), axis=1).values

X_train_word_average = word_averaging_list(wv,train_tokenized)
X_test_word_average = word_averaging_list(wv,test_tokenized)

# Compute Accuracy
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(n_jobs=1, C=1e5)
logreg = logreg.fit(X_train_word_average, train['Genre'])
y_pred = logreg.predict(X_test_word_average)
print('accuracy %s' % accuracy_score(y_pred, test.tags))
print(classification_report(test.tags, y_pred,target_names=my_tags))
