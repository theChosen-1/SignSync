import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import DeepNeuralNetwork
from src.utils import load_data
import numpy as np

print("="*60)
print("LEARNING RATE SEARCH")
print("="*60)

train_x, train_y, test_x, test_y, classes = load_data()

# Test range of learning rates
learning_rates = [0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03]
results = []

for lr in learning_rates:
    print(f"\n[Testing LR={lr}]")
    model = DeepNeuralNetwork([784, 128, 64, 24], learning_rate=lr)
    
    # Train for 5000 iterations (faster than 10K)
    model.train(train_x, train_y, num_iterations=5000, print_cost=False)
    
    train_acc = model.evaluate(train_x, train_y)
    test_acc = model.evaluate(test_x, test_y)
    gap = train_acc - test_acc
    final_cost = model.costs[-1] if model.costs else None
    
    results.append({
        'lr': lr,
        'train': train_acc,
        'test': test_acc,
        'gap': gap,
        'final_cost': final_cost
    })
    
    print(f"  Train: {train_acc:.2%} | Test: {test_acc:.2%} | Gap: {gap:.2%} | Cost: {final_cost:.4f}")

# Find best test accuracy
best = max(results, key=lambda x: x['test'])

print(f"\n{'='*60}")
print("RESULTS SUMMARY")
print(f"{'='*60}")
print(f"{'LR':<8} {'Test':<8} {'Train':<8} {'Gap':<8} {'Cost':<8}")
print("-"*60)
for r in sorted(results, key=lambda x: x['test'], reverse=True):
    marker = " ← BEST" if r == best else ""
    print(f"{r['lr']:<8.4f} {r['test']:<8.2%} {r['train']:<8.2%} {r['gap']:<8.2%} {r['final_cost']:<8.4f}{marker}")

print(f"\n✓ Best Learning Rate: {best['lr']}")
print(f"✓ Best Test Accuracy: {best['test']:.2%}")
print(f"✓ Train-Test Gap: {best['gap']:.2%}")
