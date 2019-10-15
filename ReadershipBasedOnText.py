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
import numpy as np

import nltk
from nltk.corpus import brown
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('brown')
word_list = brown.words()
word_set = set(word_list)
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Importing the data ----------------------------------------------------------
print("Importing data...")
books = json.load(gzip.open(r'C:\Users\Eli\Desktop\Python\THR\books.json.gz', "rt", encoding="utf-8"))
df = pd.DataFrame(books)
pd.set_option('display.max_colwidth', -1)

#remove all non english books
df = df[df.language == 'en']

book_text, text_len, num_words, misspellings, num_books_by_author = [], [], [], [], []

# Making a df with ALL the variables ------------------------------------------
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


# Making the actual df I am going to use --------------------------------------
abv75p, abv50p, abv25p, bottom25p = [],[],[],[]

p75 = np.percentile(df['num_ratings'], 75)
p50 = np.percentile(df['num_ratings'], 50)
p25 = np.percentile(df['num_ratings'], 25)

for row in df.itertuples():
    bottom25p.append(int(row.num_ratings <= p25))
    abv25p.append(int(row.num_ratings > p25 and row.num_ratings <= p50))
    abv50p.append(int(row.num_ratings > p50 and row.num_ratings <= p75))
    abv75p.append(int(row.num_ratings > p75))

df.drop(df.columns.difference(['text','num_ratings']), 1, inplace=True) #include at least two columns so its not a series
df = df.assign(abv75p=abv75p)
df = df.assign(abv50p=abv50p)
df = df.assign(abv25p=abv25p)
df = df.assign(bottom25p=bottom25p)
df = df.drop(['num_ratings'], axis=1)

# Training the model ----------------------------------------------------------
print("Training the model...")

train, test = train_test_split(df, test_size=0.33, shuffle=True)
X_train = train.text
X_test = test.text

stop_words = set(stopwords.words('english'))

SVC_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words=stop_words)),
            ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1)),
        ])
    
percentiles = np.array((df.columns.values)[1:])

totalAcc = 0
i = 0

AI_Assigned_Percentile = []

for percentile in percentiles:
    print('... Processing {}'.format(percentile))
   
    #train the model using X_dtm & y
    SVC_pipeline.fit(X_train, train[percentile])
    
    #compute the testing accuracy
    prediction = SVC_pipeline.predict(X_test)
    print('Test accuracy is {}'.format(accuracy_score(test[percentile], prediction)))
    acc = accuracy_score(test[percentile], prediction)
    
    totalAcc += acc 
    i += 1
    if i >= 4:
        totalAcc = totalAcc / 4
        print("\n")
        print('Overall average test accuracy for percentile predictions is {}'.format(totalAcc))
        totalAcc = 0

    finalPrediction = (SVC_pipeline.predict(df.text))
    j = 0
    for elem in finalPrediction:
        if(j >= len(AI_Assigned_Percentile)):
            AI_Assigned_Percentile.append([])
        if (elem):
            AI_Assigned_Percentile[j].append(percentile)
        j += 1
