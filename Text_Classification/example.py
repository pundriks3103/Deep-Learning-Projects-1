#Importing the required libraries

import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
import nltk
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import one_hot

#Loading the Dataset

df = pd.read_csv("/content/drive/My Drive/Colab Notebooks/Mylo/train.csv", encoding = 'unicode_escape')
df = df.drop('post_id', axis = 1)
#df.head()

#Creating input and output feature Columns

corpus = df.iloc[:,0].values
y = df.iloc[:,2].values

labelencoder = LabelEncoder()
onehotencoder = OneHotEncoder()
y = labelencoder.fit_transform(y)
y = y.reshape(-1,1)
y = onehotencoder.fit_transform(y).toarray()

#Using Word Embedding Method

nltk.download('punkt')
all_words = []
for sent in corpus:
    tokenize_word = word_tokenize(sent)
    for word in tokenize_word:
        all_words.append(word)

unique_words = set(all_words)
print(len(unique_words))

vocab_length = len(unique_words)

embedded_sentences = [one_hot(sent, vocab_length) for sent in corpus]
print(embedded_sentences )

word_count = lambda sentence: len(word_tokenize(sentence))
longest_sentence = max(corpus, key=word_count)
length_long_sentence = len(word_tokenize(longest_sentence))

padded_sentences = pad_sequences(embedded_sentences, length_long_sentence, padding='post')
print(padded_sentences)

#Adding Column *'user_stage'* in the input features *'padded_sentences'

df_temp=pd.DataFrame(data=padded_sentences[0:,0:], index=[i for i in range(padded_sentences.shape[0])], columns=['f'+str(i) for i in range(padded_sentences.shape[1])])
df_temp['f' + str(padded_sentences.shape[1])] = df['user_stage']
#df_temp.head()
X = df_temp.iloc[:, :].values
le1 = LabelEncoder()
X[:, 159] = le1.fit_transform(X[:, 159])

new_length_long_sentence = length_long_sentence + 1

#Building Model

model = Sequential()
model.add(Embedding(vocab_length, 50, input_length=new_length_long_sentence))
model.add(Flatten())
model.add(Dense(7, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
print(model.summary())

model.fit(X, y, batch_size =  10, nb_epoch = 100)

loss, accuracy = model.evaluate(X, y)
print('Accuracy: %f' % (accuracy*100))

#Predicting Results for "test dataset"

df_test = pd.read_csv("/content/drive/My Drive/Colab Notebooks/Mylo/test.csv", encoding = 'unicode_escape')
df_test = df_test.drop('post_id', axis = 1)

corpus_test = df_test.iloc[:,0].values
nltk.download('punkt')
all_words_test = []
for sent in corpus_test:
    tokenize_word_test = word_tokenize(sent)
    for word_test in tokenize_word_test:
        all_words_test.append(word_test)

embedded_sentences_test = [one_hot(sent, vocab_length) for sent in corpus_test]
print(embedded_sentences_test)

padded_sentences_test = pad_sequences(embedded_sentences_test, length_long_sentence, padding='post')
print(padded_sentences_test)

df_temp_test=pd.DataFrame(data=padded_sentences_test[0:,0:], index=[i for i in range(padded_sentences_test.shape[0])], columns=['f'+str(i) for i in range(padded_sentences_test.shape[1])])
df_temp_test['f' + str(padded_sentences_test.shape[1])] = df_test['user_stage']
X_test = df_temp_test.iloc[:, :].values
X_test[:, 159] = le1.fit_transform(X_test[:, 159])

y_pred = model.predict(X_test)
y_pred_final = np.argmax(y_pred, axis=1)

#y_pred_final

y_result = labelencoder.inverse_transform(y_pred_final)
y_result=pd.DataFrame(data = y_result)
# Loading the dataset
df1 = pd.read_csv("/content/drive/My Drive/Colab Notebooks/Mylo/test.csv", encoding = 'unicode_escape')
df1['tag'] = y_result

#df1.head()

df1.to_csv("jain.shubham102.csv")

