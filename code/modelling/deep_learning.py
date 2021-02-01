import os

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from collections import Counter
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import string

import keras
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, Dropout
from keras.layers import LSTM
from sklearn import metrics

import matplotlib.pyplot as plt
#import scikitplot as skplt
#from scikitplot import metrics


#%%
########## IMPORT DATA ##########

# import data from csv
data = pd.read_csv('../../data/twitter data/Labelled/clean/labelled_twitter_data.csv')
data = data.sample(frac=1).reset_index(drop=True)
data['tweet'] = data['tweet'].astype(str)
#data['sentiment'] = data['sentiment'].astype(str)

# split into train & test data
train = data.head(1200000)
train_labels = train['sentiment'].to_numpy()
train = train['tweet'].to_numpy()

test = data.tail(400000)
test_labels = test['sentiment'].to_numpy()
test = test['tweet'].to_numpy()



#%%
########## DATA PROCESSING ##########

# splitting data into tokens
data_train = []
data_test = []
for i in range(len(train)):
    data_i = train[i].split()
    data_train.append(data_i)
for i in range(len(test)):
    data_i = test[i].split()
    data_test.append(data_i)


# limiting vocab to 5000 words
vocab = []
for i in range(len(data_train)):
    for word in data_train[i]:
        vocab.append(word)

vocab = Counter(vocab)
vocab = [voc[0] for voc in vocab.most_common(5000)]
vocab = set(vocab)
for i in range(len(data_train)):
    data_train[i] = [w for w in data_train[i] if w in vocab]
for i in range(len(data_test)):
    data_test[i] = [w for w in data_test[i] if w in vocab]

# truncating string to max 100 words
for i in range(len(data_train)):
    data_train[i] = data_train[i][0:100]
for i in range(len(data_test)):
    data_test[i] = data_test[i][0:100]
    
# tokenizing the data
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(data_train)
train_sequences = tokenizer.texts_to_sequences(data_train)
test_sequences = tokenizer.texts_to_sequences(data_test)

# padding the data
data_train = np.array(pad_sequences(train_sequences, maxlen=100))
data_test = np.array(pad_sequences(test_sequences, maxlen=100))

# converting labels to categorical data
train_labels = to_categorical(train_labels)
train_labels = [int(i[0]) for i in train_labels]
train_labels = np.array(train_labels)
test_labels = to_categorical(test_labels)
test_labels = [int(i[0]) for i in test_labels]
test_labels = np.array(test_labels)

# splitting the training data to get 1000 training reviews
X_train, X_validate, y_train, y_validate = train_test_split(data_train, train_labels, test_size=0.75, random_state=42)

#%%
########## LOADING GloVe MODEL ##########

# importing the glove word embeddings
glove = open('../../data/tools/glove.6B.50d.txt', 'r', encoding='utf-8')
model = {}
for line in glove:
    splitline = line.split()
    word = splitline[0]
    embedding = np.array([float(val) for val in splitline[1:]])
    model[word] = embedding
    
# building an embedding matrix
word_index = tokenizer.word_index

EMBEDDING_DIM = model.get('a').shape[0]
num_words = min(10000, len(model))+1
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

for word,i in word_index.items():
    if i > 100:
        continue
    embedding_vector = model.get(word)
    if model is not None:
        embedding_matrix[i] = embedding_vector
        
# setting any nan values in the matrix to be 0        
embedding_matrix = np.nan_to_num(embedding_matrix)


#%%
########### INITALISING MODEL ###########

# initialising the model
model = Sequential()

# adding the pre-trained embedding layer
model.add(Embedding(num_words, 50, input_length=100, weights = [embedding_matrix],trainable=False))
# adding an LSTM layer
model.add(LSTM(64, return_sequences=True))
# flattening the output of the layer above
model.add(Flatten())
# adding a dense fully connected layer
model.add(Dense(64, activation='relu'))
# adding a dropout term for regularization
model.add(Dropout(0.5))
# adding the output layer
model.add(Dense(1, activation='sigmoid'))

# compiling the model
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# printing the model summary
model.summary()
keras.utils.plot_model(
    model, to_file='model.png', show_shapes=False, show_dtype=False,
    show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96
)

# fitting the model to the training data
history = model.fit(X_train, y_train, validation_data=(X_validate, y_validate), batch_size=32, epochs=25)


#%%
########## TEST & EVALUATE MODEL ##########

model.evaluate(data_test, test_labels)

probas = model.predict(data_test)
predictions = (probas > 0.5).astype(np.int)

accuracy = metrics.accuracy_score(predictions, test_labels)
precision = metrics.precision_score(predictions, test_labels)
recall = metrics.recall_score(predictions, test_labels)
f1_score = metrics.f1_score(predictions, test_labels)
auc = metrics.roc_auc_score(predictions, test_labels)
print('Random Forest - Count Vectorizer:')
print('Accuracy: ', accuracy)
print('Precision: ', precision)
print('Recall: ', recall)
print('F1 Score: ', f1_score)
print('AUC: ', auc)

# plotting the training and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epochs')
plt.legend(['training','validation'], loc='upper left')
plt.show()

 
# save model
#export_dir='./saved_model'
#tf.saved_model.save(model, export_dir=export_dir)
