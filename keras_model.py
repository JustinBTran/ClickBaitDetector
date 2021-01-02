import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

keras = tf.keras
max_features = 2000
batch_size = 64
epochs = 10

data = pd.read_csv('clickbait_data.csv')
train, val = train_test_split(data, test_size=0.2)



#Build the model (Bi-directional LSTM)

#Input for variable-length sequences of integers
inputs = keras.Input(shape=(None,), dtype="int32")
#Embed each integer in a 128-dimensional vector
x = layers.Embedding(max_features, 128)(inputs)
#Add 2 bidirectional LSTMs
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(64))(x)
#Add a classifier
outputs = layers.Dense(1, activation = "sigmoid")(x)
model = keras.Model(inputs,outputs)
#model.summary()


tokenizer = Tokenizer(num_words = max_features)
tokenizer.fit_on_texts(data['headline'])
word_index = tokenizer.word_index
print("length of dictionary is ", len(word_index))
#t_headlines = np.asarray(tokenizer.texts_to_sequences(train['headline'])).astype('float32')
#t_headlines = np.asarray([np.asarray(sentence) for sentence in t_headlines])
#t_headlines = t_headlines.astype('float32')

t_headlines = tokenizer.texts_to_sequences(train['headline'])
t_headlines = pad_sequences(t_headlines, padding='post')
t_labels = train['clickbait'].to_numpy().astype('float32')
#v_headlines = tokenizer.texts_to_sequences(val['headline'])
#v_headlines =  np.asarray([np.asarray(sentence).astype('float32') for sentence in v_headlines])
v_headlines = tokenizer.texts_to_sequences(val['headline'])
v_headlines = pad_sequences(v_headlines, padding='post')
v_labels = val['clickbait'].to_numpy().astype('float32')




model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])
history = model.fit(t_headlines,t_labels, batch_size = batch_size, epochs=epochs, validation_data=(v_headlines,v_labels))
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()