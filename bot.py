import json
import pickle
import numpy as np

import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# 1. Memuat Model dan Sumber Daya
model = load_model('path/to/file/bot_model.h5')

with open('path/to/file/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('path/to/file/label_enc.pkl', 'rb') as f:
    encoder = pickle.load(f)

with open('path/to/file/responses.json', 'r') as f:
    responses = json.load(f)

# 2. Mendefinisikan Preprocessing & Fungsi untuk Inferensi
def pprocess(txt):
    token = word_tokenize(txt)
    token = [kata for kata in token if kata not in string.punctuation]

    hapus = set(stopwords.words('indonesian'))
    saring = [kata for kata in token if kata.lower() not in hapus]

    lema = WordNetLemmatizer()
    tokens = [lema.lemmatize(kata) for kata in saring]

    return ' '.join(tokens)

def response(input_text):
    # preprocessing input
    msg = pprocess(input_text)
    
    # Tokenisasi dan konversi input_text menjadi urutan angka
    seq = tokenizer.texts_to_sequences([msg])
    padded = pad_sequences(seq)

    # Melakukan prediksi
    prediction = model.predict(padded)
    tag_index = np.argmax(prediction, axis=1)  # Ambil indeks tag dengan probabilitas tertinggi
    tag = encoder.inverse_transform(tag_index)[0]  # Konversi indeks kembali ke tag

    # Mengambil respons sesuai tag yang diprediksi
    return np.random.choice(responses[tag])  # Pilih respons secara acak dari daftar respons

# 3. Menggunakan Fungsi Inferensi ke streamlit
def chatbot(msg):
    bot = response(msg)
    return bot
