# SignSync: ASL Alphabet Recognition

A deep neural network built from scratch to recognize American Sign Language (ASL) alphabet gestures. This project demonstrates fundamental deep learning concept.

![Demo](assets/demo.gif)

## Overview

SignSync classifies static hand gestures representing 24 letters of the ASL alphabet (excludes J and Z which require motion).

## Features

- **Implementation** - Built from scratch to understand the fundamentals
- **Multi-class classification** - 24 letter classes using softmax activation
- **Configurable architecture** - Easy to experiment with different layer sizes
- **Model persistence** - Save and load trained models
- **Comprehensive evaluation** - Confusion matrices, per-class metrics, error analysis

## Architecture

INPUT → DENSE(128) → RELU → DENSE(64) → RELU → DENSE(24) → SOFTMAX

**Total Parameters:** ~

**Key Components:**
- He initialization for better convergence
- Categorical cross-entropy loss
- Mini-batch gradient descent
- Vectorized operations for efficiency

## Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/SignalNet.git
cd SignSync

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
Download Dataset
bash# Download Sign Language MNIST from Kaggle
```
