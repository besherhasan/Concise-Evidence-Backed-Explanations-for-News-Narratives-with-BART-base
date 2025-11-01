#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Step 1: Preprocessing
echo "Step 1: Preprocessing the data..."
python src/preprocess.py

# Step 2: Training
echo "Step 2: Training the model..."
python src/train.py

# Step 3: Evaluation
echo "Step 3: Evaluating the model..."
python src/evaluate.py

# Step 4: Text Generation
echo "Step 4: Generating explanations for the development set..."
python src/generate.py

echo "Workflow completed successfully!"
