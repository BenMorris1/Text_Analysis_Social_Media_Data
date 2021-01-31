import pickle
import csv
import sys
from glob import glob
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from spellchecker import SpellChecker

from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer



#%%
########## DATA EXTRACTION ##########

# create variables to locate csv files
locations = ['California','Florida','New York','Texas']
biden_path = '../../data/twitter data/Biden/'
trump_path = '../../data/twitter data/Trump/'

# create emppty list variables for csv file locations
biden_CSVs = []
trump_CSVs = []
biden_locations = []
trump_locations = []

# append file locations to list variables
for location in locations:
    # biden file paths
    path = biden_path + location + '/'
    csvs = glob(path + '*.csv')
    for file in csvs:
        biden_CSVs.append(file)
        biden_locations.append(location)
    
    # trump file paths
    path = trump_path + location + '/'
    csvs = glob(path + '*.csv')
    for file in csvs:
        trump_CSVs.append(file)
        trump_locations.append(location)

# extract data from
dates = []
tweets = []
candidates = []
locations = []
j = 0

# read biden csv files and write data to list
for csvFile in biden_CSVs:
    location = biden_locations[j]
    with open(csvFile, "r", encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=",")
        for i, line in enumerate(reader):
            try:
                if len(line[0]) == 30:
                    tweets.append(line[1])
                    dates.append(line[0][4:10] + ' ' + line[0][-4:])
                    candidates.append('Biden')
                    locations.append(location)
                else:
                    pass
            except:
                pass
    j += 1
  
j = 0
# read trump csv files and write data to list          
for csvFile in trump_CSVs:
    location = trump_locations[j]
    with open(csvFile, "r", encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=",")
        for i, line in enumerate(reader):
            try:
                if len(line[0]) == 30:
                    tweets.append(line[1])
                    dates.append(line[0][4:10] + ' ' + line[0][-4:])
                    candidates.append('Trump')
                    locations.append(location)
                else:
                    pass
            except:
                pass
    j += 1

# create dataframe from list variables
data = pd.DataFrame({'Date':dates, 'Tweet':tweets, 'Candidate':candidates, 'Location':locations})

# convert dates to datetime
#data['Date]
            
            
#%%
########## DATA PROCESSING ##########

# import files & functions for data cleaning
emoticon_dictionary = pd.read_csv('../../data/tools/emoticon_dictionary.csv')
acronym_dictionary = pd.read_csv('../../data/tools/acronym_dictionary.csv')
word_dictionary = pd.read_csv('../../data/tools/word_dictionary.csv')
stop = stopwords.words('english') 
spell = SpellChecker()
lemmatizer = WordNetLemmatizer()

# import text processign functions
sys.path.insert(0, '../data_processing')
import data_processing_functions

# apply data cleaning functions
data['emoticon_sentiment'] = data['Tweet'].apply(lambda x: data_processing_functions.extract_emoticon_sentiment(x, emoticon_dictionary))
data[['pos_emoticon', 'neg_emoticon']] = pd.DataFrame(data['emoticon_sentiment'].tolist(), index=data.index)   
data = data.drop(columns = ['emoticon_sentiment'])
data['Tweet'] = data['Tweet'].apply(lambda x: data_processing_functions.remove_emoticons(x, emoticon_dictionary))
data['Tweet'] = data['Tweet'].apply(data_processing_functions.remove_specific_language)
data['Tweet'] = data['Tweet'].apply(data_processing_functions.lowercase)
data['Tweet'] = data['Tweet'].apply(lambda x: data_processing_functions.expand_acronym(x, acronym_dictionary, acronym_dictionary['Acronym'].tolist(), acronym_dictionary['Meaning'].tolist()))
data['Tweet'] = data['Tweet'].apply(data_processing_functions.remove_punctuation)
data['Tweet'] = data['Tweet'].apply(lambda x: data_processing_functions.remove_stopwords(x, stop))
#data['Tweet'] = data['Tweet'].apply(lambda x: data_processing_functions.spelling_correction(x, spell, lemmatizer, word_dictionary))


#%%
########## FEATURE EXTRACTION ##########

# load count vect & tfidf vect
count_vect = pickle.load(open('../modelling/count_vect', 'rb'))
tfidf_transformer = pickle.load(open('../modelling/tfidf_vect', 'rb'))

data_split = [data[:5000], data[5001:10000], data[10001:15000], data[15001:20000], data[20001:25000], 
              data[25001:30000], data[30001:35000], data[35001:40000], data[40001:45000], data[45001:50000], data[50001:]]
final_data = []
x = 21

for i in range(x):
    split = 52500/x
    try:
        df = data[int(split*i):int(split*(i+1))]
    except:
        df = data[int(split*i):]

    # create count vectorizer features
    X_counts = pd.DataFrame(count_vect.transform(df['Tweet']).todense(), columns = count_vect.get_feature_names())
    
    # create tfidf vectorizer features
    X_tfidf = pd.DataFrame(tfidf_transformer.transform(X_counts).todense())
    
    # append dataframe
    #X_features = pd.concat([X_features, X_tfidf], axis=0)
    
    # concatenate data
    final_data.append(pd.concat([df[['pos_emoticon','neg_emoticon']].reset_index(drop=True), X_tfidf.reset_index(drop=True)], axis=1).fillna(0))

# concatenate dataframes
#data_tfidf = pd.concat([data[['pos_emoticon','neg_emoticon']], X_features], axis=1)

#%%
########## GENERATE PREDICTIONS ##########

# load model
model = pickle.load(open('../../data/model/logistic_regression', 'rb'))
predictions = np.array([])

for final in final_data:
    predict = model.predict(final.iloc[:,:-1])
    predictions = np.append([predictions], [predict])

# save predictions to dataframe
data['Sentiment'] = predictions


#%%
########## SAVE FINAL DATA ##########
data.to_csv('../../data/twitter data/final/data.csv')

