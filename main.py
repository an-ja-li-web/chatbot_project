import os
import sys
import nltk
import ssl


try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

print("Downloading required NLTK data...")
try:
    nltk.download('punkt_tab')
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    print("NLTK data downloaded successfully!")
except:
    print("Some NLTK data might already be installed")
from src.data_preprocessing import DataPreprocessor
from src.model_training import ChatbotModel
from src.chatbot import Chatbot

def train_chatbot():
    """Train the chatbot model"""
    print("="*50)
    print("TRAINING CHATBOT MODEL")
    print("="*50)
    
   
    os.makedirs('models', exist_ok=True)
    

    print("Step 1: Preprocessing data...")
    preprocessor = DataPreprocessor()
    intents = preprocessor.load_intents('data/intents.json')
    
    if not intents:
        print("Error: Could not load intents.json file!")
        return
    
    words, classes, documents = preprocessor.preprocess_data(intents)
    
    if not words:
        print("Error: Could not preprocess data!")
        return
    
 
    print("Step 2: Creating training data...")
    train_x, train_y = preprocessor.create_training_data()
    
 
    print("Step 3: Saving preprocessed data...")
    preprocessor.save_preprocessed_data(words, classes)
    
   
    print("Step 4: Training neural network model...")
    model = ChatbotModel(len(train_x[0]), len(train_y[0]))
    history = model.train(train_x, train_y, epochs=200, verbose=1)
    
   
    print("Step 5: Saving trained model...")
    model.save_model('models/chatbot_model.h5')
    
    print("="*50)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("="*50)
    print("You can now run the chatbot with: python main.py")
    
    return history

def run_chatbot():
    """Run the chatbot"""
    print("="*50)
    print("STARTING CHATBOT")
    print("="*50)
    
    try:
       
        chatbot = Chatbot(
            'models/chatbot_model.h5',
            'models/words.pkl',
            'models/classes.pkl',
            'data/intents.json'
        )
        
        print("="*50)
        print("Chatbot is ready! Type your message and press Enter.")
        print("Type 'quit', 'exit', or 'bye' to stop the chatbot.")
        print("="*50)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    print("Bot: Goodbye! Have a great day!")
                    break
                
                if not user_input:
                    print("Bot: Please say something!")
                    continue
                
                response = chatbot.chat(user_input)
                print(f"Bot: {response}")
                
            except KeyboardInterrupt:
                print("\nBot: Goodbye! Have a great day!")
                break
            except Exception as e:
                print(f"Bot: Sorry, I encountered an error: {e}")
                
    except FileNotFoundError as e:
        print("ERROR: Model files not found!")
        print(f"Details: {e}")
        print("\nPlease train the model first by running:")
        print("python main.py --train")
        
    except Exception as e:
        print(f"ERROR: {e}")

def show_help():
    """Show help information"""
    print("="*50)
    print("CHATBOT HELP")
    print("="*50)
    print("Available commands:")
    print("  python main.py --train    : Train the chatbot model")
    print("  python main.py --help     : Show this help message")
    print("  python main.py            : Run the chatbot")
    print("="*50)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == '--train':
            train_chatbot()
        elif sys.argv[1] == '--help':
            show_help()
        else:
            print("Unknown command. Use --help for available commands.")
    else:
        run_chatbot()
