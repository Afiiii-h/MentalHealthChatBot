import random
import json
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# Load intents and model
with open('data/intents.json') as file:
    intents = json.load(file)

model = load_model('chat_model.h5')
data = pickle.load(open("training_data.pkl", "rb"))
words = data['words']
classes = data['classes']

# Conversation memory
conversation_memory = {"last_intent": None}

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words if word.isalpha()]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [1 if w in sentence_words else 0 for w in words]
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]), verbose=0)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(sentence):
    intents_list = predict_class(sentence)
    if len(intents_list) == 0:
        return "I'm not sure I understood that."
    tag = intents_list[0]['intent']
    conversation_memory["last_intent"] = tag
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

def continue_conversation(user_input):
    """
    Handles vague or follow-up inputs using last intent.
    """
    if conversation_memory["last_intent"]:
        tag = conversation_memory["last_intent"]
        for intent in intents["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])
    return "I'm still not sure I understood. Can you clarify?"

# Main loop
print("Chatbot is ready! Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Chatbot: Take care! Goodbye 👋")
        break

    response = get_response(user_input)

    if response.startswith("I'm not sure"):
        response = continue_conversation(user_input)

    print("Chatbot:", response)