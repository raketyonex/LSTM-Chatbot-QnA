import json
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# 1. Memuat Model dan Sumber Daya
model = load_model('stver/models/tmp/bot_model.h5')

with open('stver/models/tmp/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('stver/models/tmp/label_enc.pkl', 'rb') as f:
    encoder = pickle.load(f)

with open('stver/models/tmp/responses.json', 'r') as f:
    responses = json.load(f)

# Memuat maxlen dari file
with open('stver/models/tmp/maxlen.txt', 'r') as f:
    maxlen = int(f.read().strip())

# 2. Mendefinisikan Fungsi untuk Inferensi
def response(input_text):
    # Tokenisasi dan konversi input_text menjadi urutan angka
    seq = tokenizer.texts_to_sequences([input_text])
    padded = pad_sequences(seq, maxlen=maxlen, padding='post')  # Gunakan maxlen yang dimuat

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
