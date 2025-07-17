import tensorflow as tf
import numpy as np
import os


tf.get_logger().setLevel('ERROR')


print(f"TensorFlow version: {tf.__version__}")
print("Testing TensorFlow components...")

try:
   
    test_model = tf.keras.Sequential([tf.keras.layers.Dense(1)])
    print("✅ TensorFlow imports working correctly!")
except Exception as e:
    print(f"❌ TensorFlow component error: {e}")
    exit(1)

class ChatbotModel:
    def __init__(self, input_shape, num_classes):
        """
        Initialize the ChatbotModel
        
        Args:
            input_shape (int): Size of input features
            num_classes (int): Number of output classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        
      
        self.model = self.build_model()
        
    def build_model(self):
        """Build the neural network model"""
        print("Building neural network model...")
        
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, input_shape=(self.input_shape,), activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])
 
        sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
        model.compile(
            loss='categorical_crossentropy',
            optimizer=sgd,
            metrics=['accuracy']
        )
        
        print("Model built successfully!")
        print(f"Input shape: {self.input_shape}")
        print(f"Output classes: {self.num_classes}")
        
        return model
    
    def train(self, train_x, train_y, epochs=200, batch_size=5, verbose=1):
        """
        Train the model
        
        Args:
            train_x: Training features
            train_y: Training labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        print(f"Training model for {epochs} epochs...")
        print("This may take a few minutes...")
        
        history = self.model.fit(
            train_x, train_y,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )
        
        print("Training completed!")
        return history
    
    def save_model(self, filepath):
        """
        Save the trained model
        
        Args:
            filepath: Path to save the model
        """
       
        os.makedirs('models', exist_ok=True)
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained model
        
        Args:
            filepath: Path to load the model from
        """
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def predict(self, input_data):
        """
        Make predictions using the trained model
        
        Args:
            input_data: Input features for prediction
            
        Returns:
            Prediction probabilities
        """
        return self.model.predict(input_data)
    
    def get_model_summary(self):
        """Print model summary"""
        return self.model.summary()


if __name__ == "__main__":
    print("Testing TensorFlow imports...")
    print(f"TensorFlow version: {tf.__version__}")
    
    try:
        chatbot = ChatbotModel(input_shape=100, num_classes=10)
        print("✅ Model created successfully!")
        chatbot.get_model_summary()
    except Exception as e:
        print(f"Error creating model: {e}")
