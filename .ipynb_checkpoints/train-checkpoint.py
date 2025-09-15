import json
import random
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import pickle

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Load intents
with open('data/intents.json') as file:
    intents = json.load(file)

words = []
classes = []
documents = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        tokenized = nltk.word_tokenize(pattern)
        words.extend(tokenized)
        documents.append((tokenized, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w.isalpha()]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in doc[0]]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Build model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])

# Train
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Save model and necessary objects
model.save('chat_model.h5')
pickle.dump({'words': words, 'classes': classes, 'train_x': train_x, 'train_y': train_y}, open("training_data.pkl", "wb"))

print("Training completed and model saved.")