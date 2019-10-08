# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:40:15 2019

@author: Eli
"""

# Import Dependencies ---------------------------------------------------------
print("Importing dependencies...")

import pandas as pd
import json
import gzip

import nltk
from nltk.corpus import brown
nltk.download('brown')
word_list = brown.words()
word_set = set(word_list)
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU

# Importing the Data ----------------------------------------------------------
print("Importing data...")
books = json.load(gzip.open(r'C:\Users\Eli\Desktop\Python\THR\books.json.gz', "rt", encoding="utf-8"))
df = pd.DataFrame(books)
pd.set_option('display.max_colwidth', -1)

#remove all non english books
df = df[df.language == 'en']

book_text, text_len, num_words, misspellings, num_books_by_author = [], [], [], [], []

#for the sake of efficiency
myDict = {}

author_count = df['author_id'].value_counts()

for row in df.itertuples():
    book_text.append(' '.join(x['text'] for x in row.pages)) 
    if (row.author_id not in myDict.keys()):
        myDict[row.author_id] = len(df[df['author_id'] == row.author_id])
    num_books_by_author.append(myDict[row.author_id])

for text in book_text:
    text_len.append(len(text))
    wordCounter = 0
    badSpell = 0
    for word in tokenizer.tokenize(text):
        if (word != 's' and word != 'm'):
            wordCounter += 1
            if (not word[0].isupper() and word.lower() not in word_set):
                #print(word)
                badSpell += 1
    num_words.append(wordCounter)
    misspellings.append(badSpell)
    
df = pd.DataFrame({
        'num_ratings': df.rating_count,
        'is_reviewed': df.reviewed.astype(int),
        'title': df.title,
        'author': df.author,
        'text': book_text,
        'text_len': text_len,
        'num_words': num_words,
        'misspellings': misspellings,
        'num_books_by_author': num_books_by_author,
        'avg_num_stars': df.rating_value,
        'total_stars': df.rating_total
        })

#for posterity
origonal_df = df

#now we take out all of the columns that are strings so we can do coorelations and basic ML
for column in df.columns:
    if (type(df[column][0]) == str):
        df = df.drop(column, axis=1)
        
for column in df.columns:
    print ("The coorelation between " + column + " and 'num_ratings' is... (drumroll please):")
    print (str((df['num_ratings'].corr(df[column])) * 100)[0:5] + "%!")
        

#just in case we want to try adding string variables instead of removing them
"""
for column in df.columns:
    if (type(df[column][0]) == str):
        encodableColumn += 1
        columnsToEncode.append(column)
if (encodableColumn):
    df = pd.get_dummies(df, prefix_sep="__", columns=columnsToEncode)
"""

x = df.drop('num_ratings', axis=1)
y = df['num_ratings']

min_max_scaler = preprocessing.MinMaxScaler()
x_scale = min_max_scaler.fit_transform(x)

X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(x_scale, y, test_size=0.3)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

'''
model = Sequential([
    Dense(32, activation='relu', input_shape=(7,)), #CHANGE THIS VARIABLE IF YOU DROP THINGS
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
'''


model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(7,)))
model.add(LeakyReLU(alpha=0.05))
model.add(LeakyReLU(alpha=0.05))
model.add(LeakyReLU(alpha=0.05))
model.add(LeakyReLU(alpha=0.05))
model.add(Dense(1, activation='sigmoid'))

    
# Training network ------------------------------------------------------------
print("Training network...")

model.compile(optimizer='sgd',
          loss='binary_crossentropy',
          metrics=['accuracy'])

train = model.fit(X_train, Y_train,
          batch_size=32, epochs=100,
          validation_data=(X_val, Y_val))

print("model is {}% accurate!".format(model.evaluate(X_test, Y_test)[1]))
