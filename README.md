NLP701 Assignment 2: Multilingual Characterization and Extraction of Narratives


This repository contains the implementation for Subtask C of SemEval-2025 Task 10, focusing on generating concise and evidence-supported explanations for dominant narratives in online news articles. The project employs a fine-tuned BART-base model to achieve state-of-the-art performance for this task.

Project Structure:

nlp701_assignment/
├── data/                            # Data files
│   ├── combined_train_set_with_splits.xlsx  # Training and validation data
│   ├── combined_dev_set.xlsx               # Development data
├── src/                             # Source code for the BART model with best score
│   ├── dataset.py                   # Dataset class
│   ├── preprocess.py                # Preprocessing code
│   ├── train.py                     # Training script
│   ├── evaluate.py                  # Evaluation script
│   ├── generate.py                  # Text generation script
├── Best Model Weight (Bart-Base)/       #Model weight of Bart-base Best model
│   ├── best_bart_model_assignment2/ # Fine-tuned BART-base model
│       ├── config.json
│       ├── pytorch_model.bin
│       ├── tokenizer_config.json
│       ├── vocab.json
├── requirements.txt                 # Dependencies for the project
├── README.md
├── Prediction/              prediction data documentation
├── Additional experiments/        # Contain extra experimetns but with poor models
and directories
└── run.sh                           # Script to run the complete workflow

Setup

Prerequisites:
Ensure you have the following installed on your system:

Python 3.8 or higher
pip (Python package manager)
GPU-enabled system for training (optional but recommended)

Install the required Python packages:
pip install -r requirements.txt

Place the necessary data files (combined_train_set_with_splits.xlsx and combined_dev_set.xlsx) in the data/ directory.



Workflow

1. Preprocessing
Prepare the dataset for training and evaluation by extracting contextual sentences and formatting the input.

python src/preprocess.py


2. Training
Train the BART-base model on the prepared dataset.

python src/train.py


3. Evaluation
Evaluate the model on the validation set and compute metrics such as BERTScore.

python src/evaluate.py


4. Text Generation
Generate explanations for the development set using the fine-tuned model.

python src/generate.py

5. Run All (Optional)

Run the entire workflow using the provided shell script:

bash run.sh


Results

Quantitative Results

The fine-tuned BART-base model achieved the following performance on the test set:

BERTScore Precision: 0.9085
BERTScore Recall: 0.8879
BERTScore F1: 0.8981

Qualitative Results
Examples of generated explanations are provided in the data/combined_dev_set_with_explanations.xlsx.



Model
The BART-base model was fine-tuned using the following configuration:

Learning Rate: 5 x 10^-5
Batch Size: 8
Epochs: 10
Optimizer: AdamW
Scheduler: Linear decay

Saved model files are located in models/best_bart_model_assignment2/.



Key Features

1- Contextual Sentence Extraction:

Extracts the most relevant sentences from articles using all-MiniLM-L6-v2 for cosine similarity.

2- Compact Input Formatting:

Combines dominant narrative, subnarrative, and contextual sentences in a structured input.

3- Fine-Tuned BART-base

Optimized for both performance and efficiency.



Contributions

This project was developed by Besher Hassan for NLP701 at MBZUAI. For any questions or issues, feel free to reach out to:

Email: Besher.Hassan@mbzuai.ac.ae

License
This project is for academic use only as part of the NLP701 course.