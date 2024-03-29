# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 12:46:25 2019

@author: Eli
"""
#First
setUp()
#Then, choose Your Own Adventure!

#New pipeline methods score slighly higher than my old method (roughly 95% accuracy instead of 93%)
a = newPipelines("NB")
a = newPipelines("SVC")
a = newPipelines("LogReg")
a = OldLogRegMethod()

#These two give me memory errors so I don't know what their accuracy would be!
ClassifierChain ()
BinaryRelevence ()

#Final comparative spreadsheet:
comparativeSpreadsheet():

#SOURCES:
#I used this article to help me write binary relevence and classifier chain code. Unfortunitely I keep getting a memory error when I use them.
#https://towardsdatascience.com/journey-to-the-center-of-multi-label-classification-384c40229bff
#I used this article to help me with the NewPipelines method
#https://towardsdatascience.com/multi-label-text-classification-with-scikit-learn-30714b7819c5

    
import matplotlib.pyplot as plt
import json
import gzip
import re
import logging
import pandas as pd
import numpy as np
from numpy import random

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

from bs4 import BeautifulSoup
%matplotlib inline
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
nltk.download('stopwords')
import re
import sys
import warnings

import flickrapi
from html.parser import HTMLParser
import json
import gzip
from sqlitedict import SqliteDict
import time

# From Article Code
import re
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import seaborn as sns

    
ImportLibraries()

# Importing the Data ==========================================================
print("importing data...")
books = json.load(gzip.open(r'C:\Users\Eli\Desktop\Python\THR\books.json.gz', "rt", encoding="utf-8"))
df = pd.DataFrame(books)
pd.set_option('display.max_colwidth', -1)

imageList = list(SqliteDict(r'C:\Users\Eli\Desktop\Python\THR\ImageData\imageinfo.sd').items())
    
# Putting the Data in a Format That is Easy to Work With ======================
print("formatting data...")
#create the lists I need
pages=[]
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

Human_Assigned_Genre = []

#fill the genre lists with data for each book (1 if the book in question is that genre and 0 if not)
for row in df.itertuples():
    pages.append(row.pages)
    
    genrePresent.append(int(not not row.categories)) # yes it's a double negative
    
    if (genrePresent):
        Human_Assigned_Genre.append(row.categories)
    
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

#list of text for each book    
bookList = []
for x in pages:
    bookList.append( ' '.join(y['text'] for y in x)) 

#list of image urls for each book 
urlList = []
for x in pages:
    urlList.append(' '.join(y['url'] for y in x)) 

#list of text asociated with the images of the book in question for each book
tempString = ""
imageWordList = []
i = 0
for urlGroup in urlList:
    while imageList[i][0] in urlGroup:
        for word in imageList[i][1]:
            tempString = tempString + word + " "
        i += 1
    imageWordList.append(tempString)
    tempString = ""

#remove non-english words and some camera metadata
metadata=re.compile("\S*\d+\S*|[Cc]anon|[Nn]ikon]")
nltk.download('words')
words = set(nltk.corpus.words.words())

temp = []
for i in imageWordList: 
    i = re.sub(metadata, "", i)
    i = " ".join(w for w in nltk.wordpunct_tokenize(i) \
              if w.lower() in words or not w.isalpha())
    temp.append(i)
imageWordList = temp

#take just the image titles
imageTitles = []
for i in imageWordList:
    temp = []
    i = i.split() 
    x = 0
    for elem in i:
        x += 1
        if x >= 9: #Found that most titles dont go beyond 8 words
            break
        if re.search("by|By|for|For", elem):#Title by John Doe
            break
        temp.append(elem)
        if re.search("\W", elem):#Title - John Doe
            break
    temp = " ".join(temp)
    imageTitles.append(temp)

#combine the book text and titles 
i = 0
textAndWords = []
for text in bookList:
    textAndWords.append(text + " " + imageTitles[i])

#make it one big data frame
df = pd.DataFrame(
    {'Book_Text': bookList, #change from bookList to textAndWords to add or subtract "image" words. 
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
print("cleaning text...")
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
df['Book_Text'] = df['Book_Text'].str.lower()
df['Book_Text'] = df['Book_Text'].apply(removePunct).apply(removeStopWords).apply(stemText)

genres = np.array((df.columns.values)[1:])
    
    
def comparativeSpreadsheet():

    genrePredictions = []
    
    #'p' = this variable was predicted by the ML model
        
    listOfGenres = [ 
        'pAlphabet',
        'pAnimals_and_Nature',
        'pArt_and_Music',
        'pBiographies',
        
        'pFairy_and_Folk_Tales',
        'pFiction',
        'pFoods',
        'pHealth',
    
        'pHistory',
        'pHolidays',
        'pMath_and_Science',
        'pNursery_Rhymes',
        
        'pPeople_and_Places',
        'pPoetry',
        'pRecreation_and_Leisure',
        'pSports'
    ]

    train, test = train_test_split(df, test_size=0.33, shuffle=True)
    X_train = train.Book_Text
    X_test = test.Book_Text

    totalAcc = 0
    i = 0
    
    SVC_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1)),
            ])
    for genre in genres:
        print('... Processing {}'.format(genre))
        # train the model using X_dtm & y
        SVC_pipeline.fit(X_train, train[genre])
        # compute the testing accuracy
        prediction = SVC_pipeline.predict(X_test)
        
        print('Test accuracy is {}'.format(accuracy_score(test[genre], prediction)))
        
        genrePredictions.append(SVC_pipeline.predict(df.Book_Text))
        
        acc = accuracy_score(test[genre], prediction)
        totalAcc += acc
        i += 1
        if i == 16:
            totalAcc = totalAcc / 16
            print("\n")
            print('Overall average test accuracy is {}'.format(totalAcc))
           
    Book_Text = df.Book_Text
    AI_Assigned_Genre = np.empty((len(Book_Text), 0)).tolist()

    genPointer = -1
    for row in genrePredictions:
        i = 0
        genPointer += 1
        print(listOfGenres[genPointer])
        for category in row:
            if category == 1:
                AI_Assigned_Genre[i].append(listOfGenres[genPointer])
            i+=1
            
        answers = pd.DataFrame(
                {'Book_Text': Book_Text, 
                 'Human_Assigned_Genre': Human_Assigned_Genre,
                 'AI_Assigned_Genre': AI_Assigned_Genre
                 }) 
    


# Multi-Label Classification ==================================================
def newPipelines (s): #This function made possible by Susan Li's article on TwoardDataScience.com
    train, test = train_test_split(df, test_size=0.33, shuffle=True)
    X_train = train.Book_Text
    X_test = test.Book_Text

    totalAcc = 0
    i = 0
    
    if s == 'SVC': 
        SVC_pipeline = Pipeline([
                    ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                    ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1)),
                ])
        for genre in genres:
            print('... Processing {}'.format(genre))
            # train the model using X_dtm & y
            SVC_pipeline.fit(X_train, train[genre])
            # compute the testing accuracy
            prediction = SVC_pipeline.predict(X_test)
            print('Test accuracy is {}'.format(accuracy_score(test[genre], prediction)))
            
            totalAcc += acc
            i += 1
            if i == 16:
                totalAcc = totalAcc / 16
                print("\n")
                print('Overall average test accuracy is {}'.format(totalAcc))
               
        return totalAcc


    if s == 'NB': 
        NB_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', OneVsRestClassifier(MultinomialNB(
                    fit_prior=True, class_prior=None))),
            ])
        for genre in genres:
                print('... Processing {}'.format(genre))
                # train the model using X_dtm & y
                NB_pipeline.fit(X_train, train[genre])
                # compute the testing accuracy
                prediction = NB_pipeline.predict(X_test)
                print('Test accuracy is {}'.format(accuracy_score(test[genre], prediction)))
                
                totalAcc += acc
                i += 1
                if i == 16:
                    totalAcc = totalAcc / 16
                    print("\n")
                    print('Overall average test accuracy is {}'.format(totalAcc))
                    
        return totalAcc 


    if s == 'LogReg':
        LogReg_pipeline = Pipeline([
                        ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                        ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)),
                    ])
        for genre in genres:
            print('... Processing {}'.format(genre))
            # train the model using X_dtm & y
            LogReg_pipeline.fit(X_train, train[genre])
            # compute the testing accuracy
            prediction = LogReg_pipeline.predict(X_test)
            print('Test accuracy is {}'.format(accuracy_score(test[genre], prediction)))            
            
            totalAcc += acc
            i += 1
            if i == 16:
                totalAcc = totalAcc / 16
                print("\n")
                print('Overall average test accuracy is {}'.format(totalAcc))
        
        return totalAcc        
                
 
        
    
def OldLogRegMethod ():
     # Train-Test Split =======================================================
    print("setting up a neural network...")
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df, test_size=0.33, shuffle=True)
    
    train_text = train['Book_Text']
    test_text = test['Book_Text']
    
    # TF-IDF ==================================================================
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,3), norm='l2')
    vectorizer.fit(train_text)
    vectorizer.fit(test_text)
    
    x_train = vectorizer.transform(train_text)
    y_train = train.drop(labels = ['Book_Text'], axis=1)
    
    x_test = vectorizer.transform(test_text)
    y_test = test.drop(labels = ['Book_Text'], axis=1)
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score
    from sklearn.multiclass import OneVsRestClassifier
    
    # Using LogReg pipeline for applying logistic regression and one vs rest classifier
    LogReg_pipeline = Pipeline([
                    ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=-1)),
                ])
    
    totalAcc = 0
    i = 0
    for genre in genres:
        print('**Processing {} books...**'.format(genre))
    
        # Training LR model on train data
        LogReg_pipeline.fit(x_train, train[genre])
    
        # Calculating accuracy
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

def BinaryRelevence ():
    
    # Train-Test Split =======================================================
    print("setting up a neural network...")
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df, test_size=0.33, shuffle=True)
    
    train_text = train['Book_Text']
    test_text = test['Book_Text']
    
    # TF-IDF ==================================================================
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,3), norm='l2')
    vectorizer.fit(train_text)
    vectorizer.fit(test_text)
    
    x_train = vectorizer.transform(train_text)
    y_train = train.drop(labels = ['Book_Text'], axis=1)
    
    x_test = vectorizer.transform(test_text)
    y_test = test.drop(labels = ['Book_Text'], axis=1)
    
    # using binary relevance
    from skmultilearn.problem_transform import BinaryRelevance
    from sklearn.naive_bayes import GaussianNB
    
    # initialize binary relevance multi-label classifier
    # with a gaussian naive bayes base classifier
    classifier = BinaryRelevance(GaussianNB())
    
    # train
    classifier.fit(x_train, y_train)
    
    # predict
    predictions = classifier.predict(x_test)
    
    # accuracy
    print("Accuracy = ",accuracy_score(y_test,predictions))
    print("\n")

def ClassifierChain ():
    # Train-Test Split =======================================================
    print("setting up a neural network...")
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df, test_size=0.33, shuffle=True)
    
    train_text = train['Book_Text']
    test_text = test['Book_Text']
    
    # TF-IDF ==================================================================
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,3), norm='l2')
    vectorizer.fit(train_text)
    vectorizer.fit(test_text)
    
    x_train = vectorizer.transform(train_text)
    y_train = train.drop(labels = ['Book_Text'], axis=1)
    
    x_test = vectorizer.transform(test_text)
    y_test = test.drop(labels = ['Book_Text'], axis=1)
    
    # using classifier chains
    from skmultilearn.problem_transform import ClassifierChain
    from sklearn.linear_model import LogisticRegression

    # initialize classifier chains multi-label classifier
    classifier = ClassifierChain(LogisticRegression())
    
    # Training logistic regression model on train data
    classifier.fit(x_train, y_train)
    
    # predict
    predictions = classifier.predict(x_test)
    
    # accuracy
    print("Accuracy = ",accuracy_score(y_test,predictions))
    print("\n")
