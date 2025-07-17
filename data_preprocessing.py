import nltk
import json
import numpy as np
from nltk.stem import WordNetLemmatizer
import pickle
import os

class DataPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.words = []
        self.classes = []
        self.documents = []
        self.ignore_words = ['?', '.', ',', '!']
        

        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading NLTK data...")
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
            nltk.download('omw-1.4')
        
    def load_intents(self, file_path):
        """Load intents from JSON file"""
        try:
            with open(file_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Error: {file_path} not found!")
            return None
    
    def preprocess_data(self, intents):
        """Preprocess the training data"""
        if not intents:
            return None, None, None
            
        print("Preprocessing data...")
        
        for intent in intents['intents']:
            for pattern in intent['patterns']:
                
                word_list = nltk.word_tokenize(pattern)
                self.words.extend(word_list)
                
               
                self.documents.append((word_list, intent['tag']))
                
              
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])
        
        self.words = [self.lemmatizer.lemmatize(w.lower()) for w in self.words if w not in self.ignore_words]
        self.words = sorted(list(set(self.words)))
        self.classes = sorted(list(set(self.classes)))
        
        print(f"Found {len(self.words)} unique words")
        print(f"Found {len(self.classes)} classes: {self.classes}")
        
        return self.words, self.classes, self.documents
    
    def create_training_data(self):
        """Create training data for the model"""
        print("Creating training data...")
        
        training = []
        output_empty = [0] * len(self.classes)
        
        for document in self.documents:
            bag = []
            pattern_words = document[0]
            pattern_words = [self.lemmatizer.lemmatize(word.lower()) for word in pattern_words]
            
          
            for word in self.words:
                bag.append(1) if word in pattern_words else bag.append(0)
         
            output_row = list(output_empty)
            output_row[self.classes.index(document[1])] = 1
            
            training.append([bag, output_row])
        
 
        np.random.shuffle(training)
        training = np.array(training, dtype=object)
        
        train_x = list(training[:, 0])
        train_y = list(training[:, 1])
        
        print(f"Training data created: {len(train_x)} samples")
        
        return np.array(train_x), np.array(train_y)
    
    def save_preprocessed_data(self, words, classes):
        """Save preprocessed data"""
        os.makedirs('models', exist_ok=True)
        with open('models/words.pkl', 'wb') as f:
            pickle.dump(words, f)
        with open('models/classes.pkl', 'wb') as f:
            pickle.dump(classes, f)
        
        print("Preprocessed data saved successfully!")
