# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 12:46:25 2019

@author: Elijah Wilde
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

from bs4 import BeautifulSoup
nltk.download('stopwords')
%matplotlib inline
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re
import sys
import warnings

# Importing the Data ==========================================================
books = json.load(gzip.open(r'C:\Users\Eli\Desktop\Python\THR\books.json.gz', "rt", encoding="utf-8"))
df = pd.DataFrame(books)
pd.set_option('display.max_colwidth', -1)

# Putting the Data in a Format That is Easy to Work With ======================
bookText=[]
genrePresent=[]

Alphabet = []
Animals_and_Nature = []
Art_and_Music = []
Biographies = []

Fairy_and_Folk_Tales = []
Fiction = []
Foods = []
Health = []

History = []
Holidays = []
Math_and_Science = []
Nursery_Rhymes = []

People_and_Places = []
Poetry = []
Recreation_and_Leisure = []
Sports = []

for row in df.itertuples():
    bookText.append(row.pages)
    
    genrePresent.append(int(not not row.categories)) # yes it's a double negative
    
    Alphabet.append(int(('Alph') in row.categories))
    Animals_and_Nature.append(int(('Anim') in row.categories))
    Art_and_Music.append(int(('ArtM') in row.categories))
    Biographies.append(int(('Biog') in row.categories))
    
    Fairy_and_Folk_Tales.append(int(('Fair') in row.categories))
    Fiction.append(int(('Fict') in row.categories))
    Foods.append(int(('Food') in row.categories))
    Health.append(int(('Heal') in row.categories))
    
    History.append(int(('Hist') in row.categories))
    Holidays.append(int(('Holi') in row.categories))
    Math_and_Science.append(int(('Math') in row.categories))
    Nursery_Rhymes.append(int(('Nurs') in row.categories))
    
    People_and_Places.append(int(('Peop') in row.categories))
    Poetry.append(int(('Poet') in row.categories))
    Recreation_and_Leisure.append(int(('Recr') in row.categories))
    Sports.append(int(('Spor') in row.categories))
       
tempString = ""
tempList = []
for x in bookText:
    for y in x:
        tempString = tempString + (y['text']) + " "  
    tempList.append(tempString)
    tempString = "" 

df = pd.DataFrame(
    {'Book_Text': tempList,
     'Genre_Present': genrePresent,
     
     'Alphabet': Alphabet,
     'Animals_and_Nature': Animals_and_Nature,
     'Art_and_Music': Art_and_Music,
     'Biographies': Biographies,
    
     'Fairy_and_Folk_Tales': Fairy_and_Folk_Tales,
     'Fiction': Fiction,
     'Foods': Foods,
     'Health': Health,
    
     'History': History,
     'Holidays': Holidays,
     'Math_and_Science': Math_and_Science,
     'Nursery_Rhymes': Nursery_Rhymes,
    
     'People_and_Places': People_and_Places,
     'Poetry': Poetry,
     'Recreation_and_Leisure': Recreation_and_Leisure,
     'Sports': Sports,
    })  

df = df[df.Genre_Present == 1]
df = df.drop("Genre_Present", axis=1)

# Clean text ==================================================================
def removePunct(text):
   result = re.sub(r'[?|!|\'|"|#]',r'',text)
   result = re.sub(r'[.|,|)|(|\|/]',r'',result)
   result = result.strip().replace("\n"," ")
   return result

stopWords = set(stopwords.words('english'))
reStopWords = re.compile(r"\b(" + "|".join(stopWords) + ")\\W", re.I)
def removeStopWords(text):
    global reStopWords
    return reStopWords.sub(" ", text)
    
stemmer = SnowballStemmer("english")
def stemText(text):
    result = ""
    for word in text.split():
        stem = stemmer.stem(word)
        result += stem
        result += " "
    result = result.strip()
    return result 

df['Book_Text'] = df['Book_Text'].str.lower()
df['Book_Text'] = df['Book_Text'].apply(removePunct).apply(removeStopWords)#.apply(stemText)

# Train-Test Split ============================================================
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, random_state=42, test_size=0.30, shuffle=True)
print(train.shape)
print(test.shape)

train_text = train['Book_Text']
test_text = test['Book_Text']

# TF-IDF ======================================================================
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,3), norm='l2')
vectorizer.fit(train_text)
vectorizer.fit(test_text)

x_train = vectorizer.transform(train_text)
y_train = train.drop(labels = ['Book_Text'], axis=1)

x_test = vectorizer.transform(test_text)
y_test = test.drop(labels = ['Book_Text'], axis=1)

# Multi-Label Classification ==================================================
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier

# Using pipeline for applying logistic regression and one vs rest classifier ==
LogReg_pipeline = Pipeline([
                ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=-1)),
            ])

totalAcc = 0
i = 0
for genre in genres:
    if genre != 'Text': # Getting a wierd key error
        print('**Processing {} books...**'.format(genre))
    
        # Training logistic regression model on train data
        LogReg_pipeline.fit(x_train, train[genre])
    
        # calculating test accuracy
        prediction = LogReg_pipeline.predict(x_test)
        acc = accuracy_score(test[genre], prediction)
        totalAcc += acc
        i += 1
        print('Test accuracy is {}'.format(acc))
        if i == 16:
            totalAcc = totalAcc / 16
            print("\n")
            print('Overall average test accuracy is {}'.format(totalAcc))
        print("\n")
