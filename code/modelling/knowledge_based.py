import numpy as np
import pandas as pd
import collections

# Sklearn packages
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Sentic Net packages
from senticnet.senticnet import SenticNet

########## IMPORT DATA ##########
data = pd.read_csv('../../data/twitter data/Labelled/clean/labelled_twitter_data.csv')
data = data.sample(frac=1).reset_index(drop=True)
data['tweet'] = data['tweet'].astype(str)


#%%
########## FEATURE CREATION ##########

# split text into list of tokens
data['tweet'] = data['tweet'].apply(lambda x: x.split())

# split count data
X_train, X_test, y_train, y_test = train_test_split(data[['tweet','pos_emoticon','neg_emoticon']], data['sentiment'],
                                                                            test_size=0.25, random_state=42)



#%%
########## MODELLING ##########

# logistic regression
def doc_sentiment(data):
    """
    """
    #Call SenticNet module
    sn = SenticNet()
    
    #Create positive and negative variables
    total_sentiment = 0
    
    #Calculate sentiment for all words in document
    for i in range(len(data)):
        #If words don't exist in SenticNet vocabulary they will return an error
        #We treat these words as if they have a sentiment of 0
        try:
            #Calculate sentiment of word
            sentiment = sn.polarity_value(data[i])
            #Update total sentiment
            total_sentiment += float(sentiment)
            
        except:
            None
    
    try:
        #If total sentiment = 0 division errors will occur
        #Calculate average sentiment for the document
        avg_sentiment = total_sentiment/len(data)
    except:
        avg_sentiment = 0
        
    if avg_sentiment >= 0:
        output = 1
    else:
        output = 0
    
    return output


# Run senticnet classifier on tweet tokens
X_train['tweet_sentiment'] = X_train['tweet'].apply(lambda x: doc_sentiment(x))
X_train = X_train.drop(columns=['tweet'])
X_test['tweet_sentiment'] = X_test['tweet'].apply(lambda x: doc_sentiment(x))
X_test = X_test.drop(columns=['tweet'])

# Run logistic regression model on emoticon & tweet sentiment
logistic_regression = LogisticRegression(solver='lbfgs')
logistic_regression = logistic_regression.fit(X_train, y_train)
logistic_regression_predictions = logistic_regression.predict(X_test)

#%%
########## EVALUATION ##########

# logistic regression - counts
logistic_regression_accuracy = metrics.accuracy_score(logistic_regression_predictions, y_test)
logistic_regression_precision = metrics.precision_score(logistic_regression_predictions, y_test)
logistic_regression_recall = metrics.recall_score(logistic_regression_predictions, y_test)
logistic_regression_f1 = metrics.f1_score(logistic_regression_predictions, y_test)
logistic_regression_auc = metrics.roc_auc_score(logistic_regression_predictions, y_test)
print('Knowledge Based:')
print('Accuracy: ', logistic_regression_accuracy)
print('Precision: ', logistic_regression_precision)
print('Recall: ', logistic_regression_recall)
print('F1 Score: ', logistic_regression_f1)
print('AUC: ', logistic_regression_auc)



