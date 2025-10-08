# scripts/evaluate.py
import numpy as np
import sys
sys.path.append('.')
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


def plot_confusion_matrix(y_true, y_pred, classes, save_path='assets/confusion_matrix.png'):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten())
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate ASL classifier')
    parser.add_argument('--model', type=str, 
                       default='models/saved_models/asl_classifier.pkl',
                       help='Path to saved model')
    args = parser.parse_args()
    
    print("="*50)
    print("Evaluating ASL Classifier")
    print("="*50)
    
    # Load data
    print("\nLoading data...")
    train_x, train_y, test_x, test_y, classes = load_data()
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = DeepNeuralNetwork.load(args.model)
    
    # Predictions
    print("\nMaking predictions...")
    train_pred = model.predict(train_x)
    test_pred = model.predict(test_x)
    
    # Accuracies
    train_acc = np.mean(train_pred == train_y)
    test_acc = np.mean(test_pred == test_y)
    
    print(f"\n{'='*50}")
    print("RESULTS")
    print(f"{'='*50}")
    print(f"Train Accuracy: {train_acc:.2%}")
    print(f"Test Accuracy:  {test_acc:.2%}")
    print(f"{'='*50}")
    
    # Confusion matrix
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(test_y, test_pred, classes)
    
    # Per-class report
    print("\nPer-class Performance:")
    print(classification_report(test_y.flatten(), test_pred.flatten(), 
                                target_names=classes, zero_division=0))

if __name__ == "__main__":
    main()