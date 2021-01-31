########## IMPORTS ##########

import time
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
from spellchecker import SpellChecker

import data_processing_functions


########## DATA ##########

# Twitter Data
data = pd.read_csv('../../data/twitter data/Labelled/raw/labelled_twitter_data.csv',encoding='latin-1')
data = data.sample(frac=1).reset_index(drop=True)
data = data.tail(1000)

# Emoticon Dictionary
emoticon_dictionary = pd.read_csv('../../data/tools/emoticon_dictionary.csv')

# Acronym Dictionary
acronym_dictionary = pd.read_csv('../../data/tools/acronym_dictionary.csv')

# import word dictionary
word_dictionary = pd.read_csv('../../data/tools/word_dictionary.csv')

# Stopwords Dictionary
stop = stopwords.words('english') 

# Spell Checker & Lemmatizer
spell = SpellChecker()
lemmatizer = WordNetLemmatizer()


########## PROCESS DATA ##########

# Start Timer
start = time.time()

# Extract Emoticon Sentiment Function
data['emoticon_sentiment'] = data['tweet'].apply(lambda x: data_processing_functions.extract_emoticon_sentiment(x, emoticon_dictionary))
data[['pos_emoticon', 'neg_emoticon']] = pd.DataFrame(data['emoticon_sentiment'].tolist(), index=data.index)   
data = data.drop(columns = ['emoticon_sentiment'])
emoticon_sentiment_time = time.time()
print('Emoticon Sentiment: ', emoticon_sentiment_time - start)

# Remove Emoticons Function
data['tweet'] = data['tweet'].apply(lambda x: data_processing_functions.remove_emoticons(x, emoticon_dictionary))
remove_emoticon_time = time.time()
print('Remove Emoticon: ', remove_emoticon_time - emoticon_sentiment_time)

# Remove Specific Language Function
data['tweet'] = data['tweet'].apply(data_processing_functions.remove_specific_language)
remove_specific_language_time = time.time()
print('Remove Specific Language: ', remove_specific_language_time - remove_emoticon_time)

# Lowercase Function
data['tweet'] = data['tweet'].apply(data_processing_functions.lowercase)
lowercase_time = time.time()
print('Lowercase: ', lowercase_time - remove_specific_language_time)

# Expand Acronym Function
data['tweet'] = data['tweet'].apply(lambda x: data_processing_functions.expand_acronym(x, acronym_dictionary, acronym_dictionary['Acronym'].tolist(), acronym_dictionary['Meaning'].tolist()))
expand_acronym_time = time.time()
print('Expand Acronym: ', expand_acronym_time - lowercase_time)

# Remove Punctuation Function
data['tweet'] = data['tweet'].apply(data_processing_functions.remove_punctuation)
remove_punctuation_time = time.time()
print('Remove Punctuation: ', remove_punctuation_time - expand_acronym_time)

# Remove Stopwords Function
data['tweet'] = data['tweet'].apply(lambda x: data_processing_functions.remove_stopwords(x, stop))
remove_stopwords_time = time.time()
print('Remove Stopwords: ', remove_stopwords_time - remove_punctuation_time)

# Spelling Correction Function
#data['tweet'] = data['tweet'].apply(lambda x: data_processing_functions.spelling_correction(x, spell, lemmatizer, word_dictionary))
#spelling_correction_time = time.time()
#print('Spelling Correction: ', spelling_correction_time - remove_stopwords_time)

# Update Sentiment Value
data['sentiment'] = data['sentiment'].replace(4,1)

# Output Total Time
print('Total Time: ', time.time() - start)

# save to csv
#data.to_csv('../../data/twitter data/Labelled/clean/labelled_twitter_data.csv', index=False)


