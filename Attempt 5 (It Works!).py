# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 06:48:47 2019

@author: Elijah Wilde
"""
# Import Dependencies ---------------------------------------------------------
print("Importing dependencies...")
import numpy as np
import re
import pandas as pd
import json
import gzip
from sqlitedict import SqliteDict

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
stop_words = set(stopwords.words('english'))
nltk.download('words')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Summary ---------------------------------------------------------------------
# This program takes in data about a book's text and genres and returns a df containing:
Human_Assigned_Genre = []
AI_Assigned_Genre = []
Book_Text = []

# Importing the Data ----------------------------------------------------------
print("Importing data...")
books = json.load(gzip.open(r'C:\Users\Eli\Desktop\Python\THR\books.json.gz', "rt", encoding="utf-8"))
df = pd.DataFrame(books)
pd.set_option('display.max_colwidth', -1)
imageList = list(SqliteDict(r'C:\Users\Eli\Desktop\Python\THR\ImageData\imageinfo.sd').items())

# Formatting the Data ---------------------------------------------------------
print("formatting data...")

genreLists = {'Alph':[], 'Anim':[], 'ArtM':[], 'Biog':[],
              'Fair':[], 'Fict':[], 'Food':[], 'Heal':[],
              'Hist':[], 'Holi':[], 'Math':[], 'Nurs':[],
              'Peop':[], 'Poet':[], 'Recr':[], 'Spor':[]}
pages = []

for row in df.itertuples():
    #if the book in question does not have any listed genre we can't use it for training
    if (row.categories):
        pages.append(row.pages)
        Human_Assigned_Genre.append(row.categories)
        for key in genreLists:
            genreLists[key].append(int(key in row.categories))
            
#list of text for each book    
for x in pages:
    Book_Text.append( ' '.join(y['text'] for y in x)) 

#list of image urls for each book 
urlList = []
for x in pages:
    temp = []
    for y in x:
        temp.append(y['url'])
    urlList.append(temp)
    
#add the words asociated with the images to the book text
metadata = re.compile("\S*\d+\S*|[Cc]anon|[Nn]ikon]")
words = set(nltk.corpus.words.words())

def Convert(tup, di): 
    for a, b in tup: 
        di.setdefault(a, []).append(b) 
    return di 

imageList = Convert(imageList, {})

#for each sentance asociated with an image url, 
#if that sentance meets certain criteria,
#clean it up and add it to the correct book's text
i = 0
for urlGroup in urlList:
    for url in urlGroup:
        if (url in imageList.keys()):
                for sentances in imageList[url]:
                    for sentance in sentances:
                        if (len(sentance) < 20):
                            sentance = " ".join(w for w in nltk.wordpunct_tokenize(sentance) \
                                                if w.lower() in words or not w.isalpha())
                            sentance = re.sub(metadata, "", sentance)
                            if len(sentance) > 1:
                                Book_Text[i] += (" " + sentance)
    i += 1

#make it a data frame for conveniance
genreLists = pd.DataFrame(genreLists)
genreLists.insert(0, 'Book_Text', Book_Text)

# Cleaning the Text -----------------------------------------------------------
print("Cleaning text...")
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

#apply text cleaning methods
genreLists['Book_Text'] = genreLists['Book_Text'].str.lower().apply(removePunct).apply(removeStopWords).apply(stemText)

# Training The Model ----------------------------------------------------------
train, test = train_test_split(genreLists, test_size=0.33, shuffle=True)
X_train = train.Book_Text
X_test = test.Book_Text

SVC_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words=stop_words)),
            ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1)),
        ])
    
genres = np.array((genreLists.columns.values)[1:])

totalAcc = 0
i = 1

for genre in genres:
    print('... Processing {}'.format(genre))
   
    #train the model using X_dtm & y
    SVC_pipeline.fit(X_train, train[genre])
    
    #compute the testing accuracy
    prediction = SVC_pipeline.predict(X_test)
    print('Test accuracy is {}'.format(accuracy_score(test[genre], prediction)))
    acc = accuracy_score(test[genre], prediction)
    totalAcc += acc
    i += 1
    if i > 16:
        totalAcc = totalAcc / 16
        print("\n")
        print('Overall average test accuracy is {}'.format(totalAcc))
    
    #add the genre predictions to the AI_Predicted_Genres list
    finalPrediction = (SVC_pipeline.predict(genreLists.Book_Text))
    j = 0
    for elem in finalPrediction:
        if(j >= len(AI_Assigned_Genre)):
            AI_Assigned_Genre.append([])
        if (elem):
            AI_Assigned_Genre[j].append(genre)
        j += 1

# The Final Result ------------------------------------------------------------
FINAL_RESULT = pd.DataFrame(
        {'Book_Text': Book_Text, 
         'Human_Assigned_Genre': Human_Assigned_Genre,
         'AI_Assigned_Genre': AI_Assigned_Genre
         }) 
