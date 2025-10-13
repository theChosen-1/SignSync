import numpy as np
import matplotlib.pyplot as plt
import sys
import os
# Add project root and src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import DeepNeuralNetwork
from src.utils import load_data

def main(): 
    print("="*50)
    print("SignSync: ASL Alphabet Recognition")
    print("="*50)
    
    # Load data
    print("\n[1/5] Loading data...")
    train_x, train_y, test_x, test_y, classes = load_data()
    print(f"✓ Loaded {train_x.shape[1]} training samples")
    print(f"✓ Loaded {test_x.shape[1]} test samples")
    print(f"✓ Classes: {len(classes)} letters (A-Y, no J/Z)")
    
    # Define architecture
    n_x = train_x.shape[0]  # 784 (28x28)
    layer_dims = [n_x, 128, 64, 24]
    
    print(f"\n[2/5] Creating model...")
    print(f"Architecture: {layer_dims}")
    model = DeepNeuralNetwork(layer_dims, learning_rate=0.0075)
    
    # Train
    print(f"\n[3/5] Training...")
    parameters, costs = model.train(
        train_x, train_y, 
        num_iterations=10000, 
        print_cost=True
    )
    
    # Evaluate
    print(f"\n[4/5] Evaluating...")
    train_acc = model.evaluate(train_x, train_y)
    test_acc = model.evaluate(test_x, test_y)
    
    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"Train Accuracy: {train_acc:.2%}")
    print(f"Test Accuracy:  {test_acc:.2%}")
    print(f"{'='*50}")
    
    # Save model
    print(f"\n[5/5] Saving model...")
    model.save('models/saved_models/asl_classifier.pkl')
    
    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(costs)
    plt.ylabel('Cost')
    plt.xlabel('Iterations (x100)')
    plt.title(f'Training Progress (LR={model.learning_rate})')
    plt.grid(True)
    plt.savefig('assets/training_curve.png', dpi=300, bbox_inches='tight')
    print("✓ Training curve saved to assets/training_curve.png")
    
    print("\n Training complete!")

if __name__ == "__main__":
    main()
