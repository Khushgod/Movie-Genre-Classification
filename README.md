# Movie-Genre-Classification

README: Movie Genre Classification Project

Project Overview
This project classifies movie genres using a neural network trained on movie descriptions. The model leverages TF-IDF text features and a Keras-based neural network for classification. The dataset includes movie titles, genres, and descriptions.

Key Features
Text Processing: TF-IDF vectorization with n-gram support.
Deep Learning: A neural network with two hidden layers and softmax output.
Evaluation: Accuracy, confusion matrix, precision-recall analysis, and visualizations.
Multi-Class Classification: Predicts 27 distinct movie genres.

Dataset
Source: train_data.txt (54,214 samples) and test_data.txt.

Columns:
title: Movie title.
genres: Movie genre (27 unique labels, e.g., drama, comedy, horror).
description: Movie plot summary.
Sample Genres: Drama (most frequent), documentary, comedy, horror, etc.

Requirements
Python 3.7+
Libraries:
bash
pandas numpy scikit-learn tensorflow scikeras matplotlib seaborn
Note: Dependency conflicts may arise (e.g., tensorflow-text, pandas versions). Use a virtual environment.

Installation
Clone the repository.
Install dependencies:
bash
pip install pandas numpy scikit-learn tensorflow scikeras matplotlib seaborn
Workflow

Data Loading:
Load train_data.txt and test_data.txt using a custom parser.
Extract descriptions (X) and genres (y).

Preprocessing:
Encode genres using LabelEncoder.
Convert text to TF-IDF features (TfidfVectorizer).

Model Architecture:
python
Model: "model"
Input Layer (Dense) → 128 neurons (ReLU)
Hidden Layer → 64 neurons (ReLU)
Output Layer → 27 neurons (Softmax)
Optimizer: Adam.
Loss: Sparse Categorical Crossentropy.

Training:
Split training data into train/validation sets (80/20).
Train for 15 epochs with a batch size of 32.

Evaluation:
Validation Accuracy: ~52%.
Overall Performance
Accuracy: 0.52 (52%) - 
reasonable for a 27-class problem.

Strengths
Western: 0.73 F1-score (precision: 0.74, recall: 0.71) - Excellent performance
Documentary: 0.72 F1-score (precision: 0.69, recall: 0.75) - Very strong
Game-show: 0.63 F1-score (precision: 0.70, recall: 0.57) - Good precision
Drama: 0.58 F1-score (precision: 0.57, recall: 0.60) - Consistent performance
Horror: 0.51 F1-score (precision: 0.51, recall: 0.52) - Balanced precision and recall

Weaknesses
Biography: 0.00 F1-score - Complete failure to identify this genre
Fantasy: 0.05 F1-score - Very poor performance
History: 0.03 F1-score - Nearly undetectable by the model
Crime: 0.09 F1-score - Struggles significantly
Romance: 0.10 F1-score - Poor detection rate

Class Performance Patterns
Well-identified genres tend to have distinctive language/themes (Documentaries, Westerns, Horror)
Poorly-identified genres often overlap with other categories (Biography overlaps with Drama, Fantasy with Sci-fi)
Underrepresented classes generally perform worse (Biography: 61 samples, History: 45 samples)
Confusion Matrix: Highlights misclassifications (e.g., drama vs. documentary).
Class Distribution: Drama and documentary dominate the dataset

Results
Validation Accuracy: 51.97%.

Key Metrics:
High precision/recall for frequent classes (e.g., documentary: precision=0.69, recall=0.75).
Poor performance on rare classes (e.g., biography, fantasy).

Visualizations:
Confusion matrix (top 15 classes).
Class distribution and precision-recall plots.
