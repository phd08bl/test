# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 16:25:22 2018

@author: 9888120
"""

from bs4 import BeautifulSoup
from urllib.request import urlopen
import numpy as np
import pandas as pd
import re
import nltk.data
from nltk.tokenize import word_tokenize


df = pd.read_csv('allstatements.csv')
df.columns = ['date','statement']
df['date'] = pd.to_datetime(df['date'])
df = df.dropna()
statement =  df.statement.tolist()

from nltk.corpus import stopwords

stops = set(stopwords.words('english'))


from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')
tokenizer.tokenize(statement[0])




tokenizer = nltk.data.load('english.pickle')
tokenizer.tokenize(statement[0])
word_tokenize(statement[0])

from nltk.stem.porter import PorterStemmer
port_stemmer = PorterStemmer()
port_stemmer.stem('maximum')

from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

text = 'My dogs is a dog.'


word_S = []
for words in word_tokenize(statement[0]):
    word_S.append(port_stemmer.stem(words))



statement_S = []
#i = 0
for line in statement:
    current_segment = tokenizer.tokenize(line)
    word_S = []
    for words in current_segment:
        word_S.append(port_stemmer.stem(words))
    statement_S.append(word_S)


df_content = pd.DataFrame({'statement_S':statement_S})

contents = df_content.statement_S.values.tolist()


contents_clean = []
all_words = []

for line in contents:
    line_clean = []
    for word in line:
        if word.lower() in stops:
            continue
        line_clean.append(word)
        all_words.append(str(word))
    contents_clean.append(line_clean)

clean_all_words = []
for line in contents_clean:
    for word in line:
        clean_all_words.append(str(word))


clean_all_words = pd.DataFrame({'clean_all_words':clean_all_words})
clean_all_words.head()

df_content_clean = pd.DataFrame({'contents_clean':contents_clean})
df_content_clean.head()

df_all_words = pd.DataFrame({'all_words':all_words})
df_all_words.head()


words_count = df_content_clean.groupby(by = ['contents_clean'])['contents_clean'].agg({"count":numpy.size})
words_count = words_count.reset_index().sort_values(by = ['count'], ascending = False)
words_count.head()



words_count = df_all_words.groupby(by = ['all_words'])['all_words'].agg({"count":numpy.size})
words_count = words_count.reset_index().sort_values(by = ['count'], ascending = False)
words_count.head()


def get_words_count(df_words, colname):
    words_count = df_words.groupby(by = [colname])[colname].agg({"count":numpy.size})
    words_count = words_count.reset_index().sort_values(by = ['count'], ascending = False)
    return words_count

df_words_count = get_words_count(clean_all_words, 'clean_all_words')



fd = nltk.FreqDist(clean_all_words.clean_all_words)
fd.plot(30,cumulative=False)