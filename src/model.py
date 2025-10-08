# src/model.py
import numpy as np
import pickle
from .utils import *

class DeepNeuralNetwork:
    """L-layer Deep Neural Network for multi-class classification."""
    
    def __init__(self, layer_dims, learning_rate=0.009):
        """
        Initialize the neural network.
        
        Parameters:
        -----------
        layer_dims : list
            Dimensions of each layer [n_x, n_h1, n_h2, ..., n_y]
        learning_rate : float
            Learning rate for gradient descent
        """
        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self.parameters = initialize_parameters_deep(layer_dims)
        self.costs = []
        
    def train(self, X, Y, num_iterations=2500, print_cost=True):
        """
        Train the neural network.
        
        Parameters:
        -----------
        X : numpy array
            Training data of shape (n_x, m)
        Y : numpy array
            Labels of shape (1, m)
        num_iterations : int
            Number of training iterations
        print_cost : bool
            Print cost every 100 iterations
            
        Returns:
        --------
        parameters : dict
            Trained parameters
        costs : list
            Cost history
        """
        for i in range(num_iterations):
            # Forward propagation
            AL, caches = L_model_forward(X, self.parameters)
            
            # Compute cost
            cost = compute_cost(AL, Y)
            
            # Backward propagation
            grads = L_model_backward(AL, Y, caches)
            
            # Update parameters
            self.parameters = update_parameters(
                self.parameters, grads, self.learning_rate
            )
            
            # Record cost
            if i % 100 == 0:
                self.costs.append(cost)
                if print_cost:
                    print(f"Cost after iteration {i}: {cost:.6f}")
        
        return self.parameters, self.costs
    
    def predict(self, X):
        """
        Make predictions on input data.
        
        Parameters:
        -----------
        X : numpy array
            Input data of shape (n_x, m)
            
        Returns:
        --------
        predictions : numpy array
            Predicted class labels
        """
        AL, _ = L_model_forward(X, self.parameters)
        predictions = np.argmax(AL, axis=0).reshape(1, -1)
        return predictions
    
    def evaluate(self, X, Y):
        """
        Calculate accuracy on dataset.
        
        Parameters:
        -----------
        X : numpy array
            Input data
        Y : numpy array
            True labels
            
        Returns:
        --------
        accuracy : float
            Classification accuracy
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == Y)
        return accuracy
    
    def save(self, filepath):
        """Save model parameters to file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'parameters': self.parameters,
                'layer_dims': self.layer_dims,
                'learning_rate': self.learning_rate,
                'costs': self.costs
            }, f)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load model from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        model = cls(data['layer_dims'], data['learning_rate'])
        model.parameters = data['parameters']
        model.costs = data['costs']
        print(f"Model loaded from {filepath}")
        return model