### STEP 1: load dataset dan preprocess ###

# import dataset json
import json

with open('stver/models/data.json', 'r') as dataset:
    data = json.load(dataset)

# ambil semua data dari dataset
tags = []
patterns = []
responses = {}
for intent in data['intents']:
    responses[intent['tag']]=intent['responses']
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# tokenisasi data
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(patterns)
vocab = len(tokenizer.word_index)  # jumlah kata unik dalam tokenizer

# konversi teks jadi urutan angka & kasih padding untuk urutan angka
from tensorflow.keras.preprocessing.sequence import pad_sequences

txt2seq = tokenizer.texts_to_sequences(patterns)
maxlen = max(len(seq) for seq in txt2seq)  # panjang maksimum urutan
x = pad_sequences(txt2seq, maxlen=maxlen, padding='post')

# buat label encoding output
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y = encoder.fit_transform(tags)


### STEP 2: pembuatan & pelatihan model ###

# import library untuk pembuatan model
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, LayerNormalization, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Membuat model Sequential
model = Sequential()

# Menambahkan lapisan-lapisan ke dalam model
model.add(Input(shape=(x.shape[1],)))  # Lapisan input dengan bentuk sesuai dengan x
model.add(Embedding(input_dim=vocab+1, output_dim=100, mask_zero=True))  # Lapisan embedding dengan vocab_size + 1 sebagai input_dim
model.add(LSTM(32, return_sequences=True))  # Lapisan LSTM dengan 32 unit dan return_sequences=True
model.add(LayerNormalization())  # Lapisan normalisasi
model.add(LSTM(32, return_sequences=True))  # Lapisan LSTM kedua dengan 32 unit dan return_sequences=True
model.add(LayerNormalization())  # Lapisan normalisasi
model.add(LSTM(32))  # Lapisan LSTM ketiga dengan 32 unit
model.add(LayerNormalization())  # Lapisan normalisasi
model.add(Dense(128, activation="relu"))  # Lapisan dense dengan 128 unit dan fungsi aktivasi ReLU
model.add(LayerNormalization())  # Lapisan normalisasi
model.add(Dropout(0.2))  # Lapisan dropout dengan tingkat dropout 0.2
model.add(Dense(128, activation="relu"))  # Lapisan dense kedua dengan 128 unit dan fungsi aktivasi ReLU
model.add(LayerNormalization())  # Lapisan normalisasi
model.add(Dropout(0.2))  # Lapisan dropout kedua dengan tingkat dropout 0.2
model.add(Dense(len(np.unique(y)), activation="softmax"))  # Lapisan dense terakhir dengan jumlah unit sesuai dengan jumlah kelas dan fungsi aktivasi softmax

# Training & mengompilasi model dengan optimizer Adam, fungsi loss sparse_categorical_crossentropy, dan metrik akurasi
model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy']) 
model.fit(x, y, batch_size=10, epochs=100, callbacks=[EarlyStopping(monitor='accuracy', patience=3)])

### STEP 3: simpan file untuk inferensi model ###

import pickle

# Menyimpan model
model.save('stver/models/tmp/bot_model.h5')

# Menyimpan tokenizer
with open('stver/models/tmp/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

# Menyimpan label encoder
with open('stver/models/tmp/label_enc.pkl', 'wb') as f:
    pickle.dump(encoder, f, protocol=pickle.HIGHEST_PROTOCOL)

# menyimpan dict response ke file json
with open('stver/models/tmp/responses.json', 'w') as f:
    json.dump(responses, f)

# Menyimpan maxlen ke dalam file
with open('stver/models/tmp/maxlen.txt', 'w') as f:
    f.write(str(maxlen))