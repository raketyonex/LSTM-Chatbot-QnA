import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
nltk.download('punkt')
nltk.download('wordnet')

from tensorflow.keras.models import load_model

import numpy as np
import json
import random
import pickle


words = pickle.load(open('models/words.pkl', 'rb'))
classes = pickle.load(open('models/classes.pkl', 'rb'))
model = load_model('models/bot_model.h5')

with open('models/data.json') as file:
    data = json.load(file)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag).reshape(1, 1, -1)

def predict_class(sentence):
    p = bow(sentence, words)
    res = model.predict(p)[0]

    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intent):
    intents_list = data['intents']
    for i in intents_list:
        if i['tag'] == intent:
            return random.choice(i['responses'])

def chatbot(message):
    intents = predict_class(message)
    intent = intents[0]['intent']
    response = get_response(intent)
    return response