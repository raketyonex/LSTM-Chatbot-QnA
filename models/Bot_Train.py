### load dataset json
import json

with open('path/to/files/data.json', 'r') as f:
    dataset = json.load(f)

# ekstrak & pisahkan data ke masing-masing variabel
responses = {}
patterns = []
tags = []
for intent in dataset['intents']:
    responses[intent['tag']] = intent['responses']
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])

import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# fungsi untuk preprocessing
def pprocess(teks):
    # tokenisasi (pecah kalimat jadi per teks) & hapus tanda baca
    token = word_tokenize(teks)
    token = [kata for kata in token if kata not in string.punctuation]

    # hapus stop word
    hapus = set(stopwords.words('indonesian'))
    saring = [kata for kata in token if kata.lower() not in hapus]

    # lematisasi (ubah kata yang kesaring ke dalam bentuk kamusnya)
    lema = WordNetLemmatizer()
    output = [lema.lemmatize(kata) for kata in saring]

    return output

## preprocessing teks dan satukan lagi jadi data teks baru
pprocess_teks = [pprocess(kata) for kata in patterns]
teks = [' '.join(tokens) for tokens in pprocess_teks]

# lanjut preprocessing data teks baru untuk pembuatan model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# buat kamus token (pecah kata dan buat jadi urutan numerik)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(teks)
vocab = len(tokenizer.word_index)

seqteks = tokenizer.texts_to_sequences(teks)
maxlen = max(len(seq) for seq in seqteks)
x = pad_sequences(seqteks, maxlen=maxlen, padding='post')

# buat label output dari tags
encoder = LabelEncoder()
y = encoder.fit_transform(tags)

### pembuatan & pelatihan model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping

# Membuat model Sequential
model = Sequential()

# Menambahkan lapisan-lapisan ke dalam model
model.add(Embedding(input_dim=vocab+1, output_dim=128, mask_zero=True, input_length=x.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(y), activation='softmax'))

# Training & mengompilasi model dengan optimizer Adam, fungsi loss sparse_categorical_crossentropy, dan metrik akurasi
model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy']) 
model.fit(x, y, batch_size=10, epochs=100, callbacks=[EarlyStopping(monitor='accuracy', patience=3)])

### impan file untuk inferensi model
import pickle

# Menyimpan model
model.save('path/to/files/bot_model.h5')

# Menyimpan tokenizer
with open('path/to/files/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

# Menyimpan label encoder
with open('path/to/files/label_enc.pkl', 'wb') as f:
    pickle.dump(encoder, f, protocol=pickle.HIGHEST_PROTOCOL)

# Menyimpan dict response ke file json
with open('path/to/files/responses.json', 'w') as f:
    json.dump(responses, f)
