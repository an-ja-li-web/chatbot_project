import nltk
import pickle
import json
import numpy as np
from tensorflow import keras
from nltk.stem import WordNetLemmatizer
import random
import os

class Chatbot:
    def __init__(self, model_path, words_path, classes_path, intents_path):
        self.lemmatizer = WordNetLemmatizer()
        
        # Check if all required files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(words_path):
            raise FileNotFoundError(f"Words file not found: {words_path}")
        if not os.path.exists(classes_path):
            raise FileNotFoundError(f"Classes file not found: {classes_path}")
        if not os.path.exists(intents_path):
            raise FileNotFoundError(f"Intents file not found: {intents_path}")
        
        # Load model and preprocessed data
        print("Loading chatbot model and data...")
        self.model = keras.models.load_model(model_path)
        self.words = pickle.load(open(words_path, 'rb'))
        self.classes = pickle.load(open(classes_path, 'rb'))
        
        # Load intents
        with open(intents_path, 'r') as file:
            self.intents = json.load(file)
        
        print("Chatbot loaded successfully!")
        print(f"Vocabulary size: {len(self.words)}")
        print(f"Number of classes: {len(self.classes)}")
    
    def clean_up_sentence(self, sentence):
        """Tokenize and lemmatize the sentence"""
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words
    
    def bag_of_words(self, sentence):
        """Create bag of words array"""
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(self.words)
        
        for s in sentence_words:
            for i, word in enumerate(self.words):
                if word == s:
                    bag[i] = 1
        
        return np.array(bag)
    
    def predict_class(self, sentence):
        """Predict the intent class"""
        bow = self.bag_of_words(sentence)
        res = self.model.predict(np.array([bow]), verbose=0)[0]
        
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        
        # Sort by probability
        results.sort(key=lambda x: x[1], reverse=True)
        
        return_list = []
        for r in results:
            return_list.append({
                'intent': self.classes[r[0]],
                'probability': str(r[1])
            })
        
        return return_list
    
    def get_response(self, intents_list):
        """Get response based on predicted intent"""
        if not intents_list:
            return "I'm sorry, I don't understand that. Can you try rephrasing?"
        
        tag = intents_list[0]['intent']
        
        for intent in self.intents['intents']:
            if intent['tag'] == tag:
                response = random.choice(intent['responses'])
                return response
        
        return "I'm sorry, I don't understand that. Can you try rephrasing?"
    
    def chat(self, message):
        """Main chat function"""
        if not message.strip():
            return "Please say something!"
        
        ints = self.predict_class(message)
        response = self.get_response(ints)
        return response
    
    def get_intent_info(self, message):
        """Get detailed intent information (for debugging)"""
        ints = self.predict_class(message)
        return ints