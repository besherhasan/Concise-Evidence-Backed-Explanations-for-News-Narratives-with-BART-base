NLP Model Experiments
This repository contains modularized implementations of various NLP models, including GPT-2 (Small, Medium, Large) and BART-Large, for tasks involving text generation and evaluation using BERTScore.

Directory Structure
Each directory contains model-specific implementations and their respective code splits.

1. Bart-Large
Description: This folder contains the implementation of the BART-Large model for conditional generation tasks.
Contents:
contextual_extraction.py: Extracts relevant sentences from articles based on narrative and sub-narrative similarity.
data_processing.py: Preprocesses the dataset and prepares input-output pairs for the model.
dataset.py: Defines the custom PyTorch dataset for text generation tasks.
model_training.py: Contains training and validation logic.
evaluation.py: Implements evaluation using BERTScore.
bart_model.py: Initializes and loads the BART-Large model and tokenizer.
main.py: Entry point script to train, validate, and evaluate the model.
2. GPT-2 Large
Description: Implements the GPT-2 Large model for narrative-based text generation.
Contents:
contextual_extraction.py: Extracts contextual sentences from articles.
data_processing.py: Processes input articles and prepares data for training/validation.
dataset.py: Defines the dataset class used for feeding data into GPT-2 Large.
model_training.py: Training loop for GPT-2 Large.
evaluation.py: Evaluates the generated text using BERTScore.
gpt_model.py: Loads the GPT-2 Large model and tokenizer.
main.py: Main script to execute all tasks.
3. GPT-2 Medium
Description: Contains the code for the GPT-2 Medium model. This implementation is more lightweight compared to GPT-2 Large.
Contents:
Similar to the GPT-2 Large folder but adapted for GPT-2 Medium.
Model-specific configurations are adjusted for the smaller model size.
4. GPT-2 Small
Description: Contains the code for the smallest variant of GPT-2. Suitable for quick experiments on smaller datasets.
Contents:
Similar structure as the other folders but uses GPT-2 Small configurations.
Dataset
The dataset used is located in data/combined_train_set_with_splits.xlsx.
The dataset is split into:
Train: Used for model training.
Validation: Used to evaluate performance during and after training.
How to Use
Install Dependencies:


pip install -r requirements.txt
Navigate to the Desired Model Folder:


cd Bart-Large
(Replace Bart-Large with GPT-2 Large, GPT-2 Medium, or GPT-2 Small for other models.)

Run the Main Script:


Copy code
python main.py
Evaluation
Each model is evaluated using BERTScore for:

Precision
Recall
F1-Score
Results are printed to the console after training and validation.

Notes
Ensure sufficient GPU resources for GPT-2 Large and BART-Large models.
For smaller models (e.g., GPT-2 Medium or Small), the training is faster and less resource-intensive.