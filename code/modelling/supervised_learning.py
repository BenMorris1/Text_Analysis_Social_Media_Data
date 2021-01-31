import numpy as np
import pandas as pd
import collections
import time
import pickle

# Sklearn packages
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# NLTK packages
from nltk.classify import accuracy
from nltk.metrics.scores import precision, recall
from nltk.classify.maxent import MaxentClassifier

########## IMPORT DATA ##########
data = pd.read_csv('../../data/twitter data/Labelled/clean/labelled_twitter_data.csv')
data = data.sample(frac=1).reset_index(drop=True)
data = data.head(9000)
data['tweet'] = data['tweet'].astype(str)

start = time.time()
#%%
########## FEATURE CREATION ##########

# create count vectorizer features
count_vect = CountVectorizer()
count_vect = count_vect.fit(data['tweet'])
X_counts = pd.DataFrame(count_vect.transform(data['tweet']).todense(), columns = count_vect.get_feature_names())
pickle.dump(count_vect, open('count_vect', 'wb'))

# create tfidf vectorizer features
tfidf_transformer = TfidfTransformer()
tfidf_transformer = tfidf_transformer.fit(X_counts)
X_tfidf = pd.DataFrame(tfidf_transformer.transform(X_counts).todense())
pickle.dump(tfidf_transformer, open('tfidf_vect', 'wb'))

# concatenate dataframes
data_count = pd.concat([data[['pos_emoticon','neg_emoticon']], X_counts], axis=1)
data_tfidf = pd.concat([data[['pos_emoticon','neg_emoticon']], X_tfidf], axis=1)


# split count data
X_train_count, X_test_count, y_train_count, y_test_count = train_test_split(data_count, data['sentiment'],
                                                                            test_size=0.25, random_state=42)

# split tfidf data
X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(data_tfidf, data['sentiment'],
                                                                            test_size=0.25, random_state=42)

# create list of tuples (nltk algorithm)
#X_train_count_dict = X_train_count.to_dict(orient="records")
#X_train_tfidf_dict = X_train_tfidf.to_dict(orient="records")
#nltk_train_counts = []
#nltk_train_tfidf = []
#for i in range(len(X_train_count)):
#    nltk_train_counts.append((X_train_count_dict[i],y_train_count.iloc[i]))
#    nltk_train_tfidf.append((X_train_tfidf_dict[i],y_train_tfidf.iloc[i]))
#    
#X_test_count_dict = X_test_count.to_dict(orient="records")
#X_test_tfidf_dict = X_test_tfidf.to_dict(orient="records")
#nltk_test_counts = []
#nltk_test_tfidf = []
#for i in range(len(X_test_count)):
#    nltk_test_counts.append((X_test_count_dict[i],y_test_count.iloc[i]))
#    nltk_test_tfidf.append((X_test_tfidf_dict[i],y_test_tfidf.iloc[i]))
#    

#%%
########## MODELLING ##########

# logistic regression
logistic_regression = LogisticRegression(solver='lbfgs')
logistic_regression_counts = logistic_regression.fit(X_train_count, y_train_count)
logistic_regression_tfidf = logistic_regression.fit(X_train_tfidf, y_train_tfidf)
logistic_regression_predictions_counts = logistic_regression_counts.predict(X_test_count)
logistic_regression_predictions_tfidf = logistic_regression_tfidf.predict(X_test_tfidf)
pickle.dump(logistic_regression_tfidf, open('logistic_regression', 'wb'))

# naive bayes
#naive_bayes = MultinomialNB()
#naive_bayes_counts = naive_bayes.fit(X_train_count, y_train_count)
#naive_bayes_tfidf = naive_bayes.fit(X_train_tfidf, y_train_tfidf)
#naive_bayes_predictions_counts = naive_bayes_counts.predict(X_test_count)
#naive_bayes_predictions_tfidf = naive_bayes_tfidf.predict(X_test_tfidf)
#
## support vector machine
#support_vector_machine = SVC(gamma='scale')
#support_vector_machine_counts = support_vector_machine.fit(X_train_count, y_train_count)
#support_vector_machine_tfidf = support_vector_machine.fit(X_train_tfidf, y_train_tfidf)
#support_vector_machine_predictions_counts = support_vector_machine_counts.predict(X_test_count)
#support_vector_machine_predictions_tfidf = support_vector_machine_tfidf.predict(X_test_tfidf)
#
## random forest
#random_forest = RandomForestClassifier()
#random_forest_counts = random_forest.fit(X_train_count, y_train_count)
#random_forest_tfidf = random_forest.fit(X_train_tfidf, y_train_tfidf)
#random_forest_predictions_counts = random_forest_counts.predict(X_test_count)
#random_forest_predictions_tfidf = random_forest_tfidf.predict(X_test_tfidf)

# maximum entropy
#max_entropy_counts = MaxentClassifier.train(nltk_train_counts, algorithm='GIS')#, n_estimators=2)
#max_entropy_tfidf = MaxentClassifier.train(nltk_train_tfidf, algorithm='GIS')#, n_estimators=2)


#%%
########## EVALUATION ##########

# logistic regression - counts
logistic_regression_accuracy_counts = metrics.accuracy_score(logistic_regression_predictions_counts, y_test_count)
logistic_regression_precision_counts = metrics.precision_score(logistic_regression_predictions_counts, y_test_count)
logistic_regression_recall_counts = metrics.recall_score(logistic_regression_predictions_counts, y_test_count)
logistic_regression_f1_counts = metrics.f1_score(logistic_regression_predictions_counts, y_test_count)
logistic_regression_auc_counts = metrics.roc_auc_score(logistic_regression_predictions_counts, y_test_count)
print('Logistic Regression - Count Vectorizer:')
print('Accuracy: ', logistic_regression_accuracy_counts)
print('Precision: ', logistic_regression_precision_counts)
print('Recall: ', logistic_regression_recall_counts)
print('F1 Score: ', logistic_regression_f1_counts)
print('AUC: ', logistic_regression_auc_counts)
print('\n')

# logistic regression - tfidf
logistic_regression_accuracy_tfidf = metrics.accuracy_score(logistic_regression_predictions_tfidf, y_test_tfidf)
logistic_regression_precision_tfidf = metrics.precision_score(logistic_regression_predictions_tfidf, y_test_tfidf)
logistic_regression_recall_tfidf = metrics.recall_score(logistic_regression_predictions_tfidf, y_test_tfidf)
logistic_regression_f1_tfidf = metrics.f1_score(logistic_regression_predictions_tfidf, y_test_tfidf)
logistic_regression_auc_tfidf = metrics.roc_auc_score(logistic_regression_predictions_tfidf, y_test_tfidf)
print('Logistic Regression - TFIDF Vectorizer:')
print('Accuracy: ', logistic_regression_accuracy_tfidf)
print('Precision: ', logistic_regression_precision_tfidf)
print('Recall: ', logistic_regression_recall_tfidf)
print('F1 Score: ', logistic_regression_f1_tfidf)
print('AUC: ', logistic_regression_auc_tfidf)
print('\n')

# naive bayes - counts
naive_bayes_accuracy_counts = metrics.accuracy_score(naive_bayes_predictions_counts, y_test_count)
naive_bayes_precision_counts = metrics.precision_score(naive_bayes_predictions_counts, y_test_count)
naive_bayes_recall_counts = metrics.recall_score(naive_bayes_predictions_counts, y_test_count)
naive_bayes_f1_counts = metrics.f1_score(naive_bayes_predictions_counts, y_test_count)
naive_bayes_auc_counts = metrics.roc_auc_score(naive_bayes_predictions_counts, y_test_count)
print('Naive Bayes - Count Vectorizer:')
print('Accuracy: ', naive_bayes_accuracy_counts)
print('Precision: ', naive_bayes_precision_counts)
print('Recall: ', naive_bayes_recall_counts)
print('F1 Score: ', naive_bayes_f1_counts)
print('AUC: ', naive_bayes_auc_counts)
print('\n')

# naive bayes - tfidf
naive_bayes_accuracy_tfidf = metrics.accuracy_score(naive_bayes_predictions_tfidf, y_test_tfidf)
naive_bayes_precision_tfidf = metrics.precision_score(naive_bayes_predictions_tfidf, y_test_tfidf)
naive_bayes_recall_tfidf = metrics.recall_score(naive_bayes_predictions_tfidf, y_test_tfidf)
naive_bayes_f1_tfidf = metrics.f1_score(naive_bayes_predictions_tfidf, y_test_tfidf)
naive_bayes_auc_tfidf = metrics.roc_auc_score(naive_bayes_predictions_tfidf, y_test_tfidf)
print('Naive Bayes - TFIDF Vectorizer:')
print('Accuracy: ', naive_bayes_accuracy_tfidf)
print('Precision: ', naive_bayes_precision_tfidf)
print('Recall: ', naive_bayes_recall_tfidf)
print('F1 Score: ', naive_bayes_f1_tfidf)
print('AUC: ', naive_bayes_auc_tfidf)
print('\n')

# support vector machine - counts
support_vector_machine_accuracy_counts = metrics.accuracy_score(support_vector_machine_predictions_counts, y_test_count)
support_vector_machine_precision_counts = metrics.precision_score(support_vector_machine_predictions_counts, y_test_count)
support_vector_machine_recall_counts = metrics.recall_score(support_vector_machine_predictions_counts, y_test_count)
support_vector_machine_f1_counts = metrics.f1_score(support_vector_machine_predictions_counts, y_test_count)
support_vector_machine_auc_counts = metrics.roc_auc_score(support_vector_machine_predictions_counts, y_test_count)
print('Support Vector Machine - Count Vectorizer:')
print('Accuracy: ', support_vector_machine_accuracy_counts)
print('Precision: ', support_vector_machine_precision_counts)
print('Recall: ', support_vector_machine_recall_counts)
print('F1 Score: ', support_vector_machine_f1_counts)
print('AUC: ', support_vector_machine_auc_counts)
print('\n')

# support vector machine - tfidf
support_vector_machine_accuracy_tfidf = metrics.accuracy_score(support_vector_machine_predictions_tfidf, y_test_tfidf)
support_vector_machine_precision_tfidf = metrics.precision_score(support_vector_machine_predictions_tfidf, y_test_tfidf)
support_vector_machine_recall_tfidf = metrics.recall_score(support_vector_machine_predictions_tfidf, y_test_tfidf)
support_vector_machine_f1_tfidf = metrics.f1_score(support_vector_machine_predictions_tfidf, y_test_tfidf)
support_vector_machine_auc_tfidf = metrics.roc_auc_score(support_vector_machine_predictions_tfidf, y_test_tfidf)
print('Support Vector Machine - TFIDF Vectorizer:')
print('Accuracy: ', support_vector_machine_accuracy_tfidf)
print('Precision: ', support_vector_machine_precision_tfidf)
print('Recall: ', support_vector_machine_recall_tfidf)
print('F1 Score: ', support_vector_machine_f1_tfidf)
print('AUC: ', support_vector_machine_auc_tfidf)
print('\n')

# random forest - counts
random_forest_accuracy_counts = metrics.accuracy_score(random_forest_predictions_counts, y_test_count)
random_forest_precision_counts = metrics.precision_score(random_forest_predictions_counts, y_test_count)
random_forest_recall_counts = metrics.recall_score(random_forest_predictions_counts, y_test_count)
random_forest_f1_counts = metrics.f1_score(random_forest_predictions_counts, y_test_count)
random_forest_auc_counts = metrics.roc_auc_score(random_forest_predictions_counts, y_test_count)
print('Random Forest - Count Vectorizer:')
print('Accuracy: ', random_forest_accuracy_counts)
print('Precision: ', random_forest_precision_counts)
print('Recall: ', random_forest_recall_counts)
print('F1 Score: ', random_forest_f1_counts)
print('AUC: ', random_forest_auc_counts)
print('\n')

# random forest - tfidf
random_forest_accuracy_tfidf = metrics.accuracy_score(random_forest_predictions_tfidf, y_test_tfidf)
random_forest_precision_tfidf = metrics.precision_score(random_forest_predictions_tfidf, y_test_tfidf)
random_forest_recall_tfidf = metrics.recall_score(random_forest_predictions_tfidf, y_test_tfidf)
random_forest_f1_tfidf = metrics.f1_score(random_forest_predictions_tfidf, y_test_tfidf)
random_forest_auc_tfidf = metrics.roc_auc_score(random_forest_predictions_tfidf, y_test_tfidf)
print('Random Forest - TFIDF Vectorizer:')
print('Accuracy: ', random_forest_accuracy_tfidf)
print('Precision: ', random_forest_precision_tfidf)
print('Recall: ', random_forest_recall_tfidf)
print('F1 Score: ', random_forest_f1_tfidf)
print('AUC: ', random_forest_auc_tfidf)
print('\n')
print('Time: ',time.time() - start)

# maximum entropy - counts
#max_ent_accuracy_counts = accuracy(max_entropy_counts, nltk_test_counts)
#
#refsets = collections.defaultdict(set)
#testsets = collections.defaultdict(set)
#
#for i, (feats, label) in enumerate(nltk_test_counts):
#    refsets[label].add(i)
#    observed = max_entropy_counts.classify(feats)
#    testsets[observed].add(i)
#    
#max_ent_precision_counts = precision(refsets['pos'], testsets['pos'])
#max_ent_recall_counts = recall(refsets['pos'], testsets['pos'])
#print('Random Forest - Count Vectorizer:')
#print('Accuracy: ', max_ent_accuracy_counts)
#print('Precision: ', max_ent_precision_counts)
#print('Recall: ', max_ent_recall_counts)
#
## maximum entropy - tfidf
#max_ent_accuracy_tfidf = accuracy(max_entropy_tfidf, nltk_test_tfidf)
#
#refsets = collections.defaultdict(set)
#testsets = collections.defaultdict(set)
#
#for i, (feats, label) in enumerate(nltk_test_tfidf):
#    refsets[label].add(i)
#    observed = max_entropy_tfidf.classify(feats)
#    testsets[observed].add(i)
#    
#max_ent_precision_tfidf = precision(refsets['pos'], testsets['pos'])
#max_ent_recall_tfidf = recall(refsets['pos'], testsets['pos'])
#print('Random Forest - Count Vectorizer:')
#print('Accuracy: ', max_ent_accuracy_tfidf)
#print('Precision: ', max_ent_precision_tfidf)
#print('Recall: ', max_ent_recall_tfidf)

