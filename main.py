import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import json
import pickle
import random

nltk.download("punkt_tab")

with open('intents.json', encoding='utf-8') as file:
    data = json.load(file)

stemmer = LancasterStemmer()

try:
    with open('data.pickle', 'rb') as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    x_docs = []
    y_docs = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            x_docs.append(wrds)
            y_docs.append(intent['tag'])

            if intent['tag'] not in labels:
                labels.append(intent['tag'])

    words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(x_docs):
        bag = []
        wrds = [stemmer.stem(w) for w in doc]
        for w in words:
            bag.append(1 if w in wrds else 0)

        output_row = out_empty[:]
        output_row[labels.index(y_docs[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    with open('data.pickle', 'wb') as f:
        pickle.dump((words, labels, training, output), f)

# Define the model
model = Sequential()
model.add(Dense(8, input_shape=(len(training[0]),), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(len(output[0]), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

try:
    model.load_weights("model.weights.h5")
except:
    model.fit(training, output, epochs=100, batch_size=8, verbose=1)
    model.save_weights("model.weights.h5")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)

def chat():
    print("The bot is ready to talk! (Type 'quit' to exit)")
    while True:
        inp = input("\nYou: ")
        if inp.lower() == 'quit':
            break

        results = model.predict(np.array([bag_of_words(inp, words)]))
        results_index = np.argmax(results)
        tag = labels[results_index]

        for tg in data['intents']:
            if tg['tag'] == tag:
                responses = tg['responses']
        print("Bot: " + random.choice(responses))

chat()
