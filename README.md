📽️ Movie Genre Classification

🧩 Project Overview
  This project focuses on predicting the genres of movies based on their plot descriptions.
  The task is framed as a multi-label classification problem since a single movie can belong to multiple genres (like "Action", "Comedy", "Drama", etc.).
  
  Features
    •	Text Processing: Uses TF-IDF vectorization with n-gram ranges (1-2) to convert movie descriptions into numerical features
    •	Neural Network: Implements a 3-layer dense neural network (128-64-output) with ReLU activation
    •	Multi-class Classification: Predicts among 27 possible movie genres
    •	Evaluation Metrics: Includes accuracy, precision, recall, and confusion matrix analysis
  
  Requirements
    *	Python 3.7+
    *	Required packages:
    * pandas
    * numpy
    *	scikit-learn
    *	tensorflow
    *	scikeras
    *	matplotlib
    *	seaborn

  Dataset
    The model uses a dataset containing data of over 5000 movies
    •	Movie titles
    •	Genre labels (27 categories)
    •	Plot descriptions
________________________________________
📊 Classification Report

'Accuracy': '53.00%', 
'Macro Average Precision': '51.00%', 
'Macro Average Recall': '49.00%', 
'Macro Average F1-Score': '48.00%', 
'Weighted Average Precision': '52.00%', 
'Weighted Average Recall': '53.00%', 
'Weighted Average F1-Score': '51.00%' 
________________________________________
🚀 How We Reached 53% Accuracy

•	Started simple: basic TF-IDF + MultiLabelBinarizer approach → 42%.
•	Introduced deep learning: using Scikeras with a properly set output layer and Binary Cross Entropy loss → 52%.
•	Enhanced features: customized TF-IDF preprocessing, added n-grams, stopword removal → 53%.
Every step involved careful tuning of both text representation and model architecture.
________________________________________
📂 Project Structure

•	Movie_Genre_Classification.ipynb → Main Jupyter Notebook containing all code and results.
•	README.md → Project documentation.
________________________________________
📚 Future Work

•	Fine-tune deep learning models with more layers (LSTM/CNN for sequence data).
•	Implement attention mechanisms for better context understanding.
•	Use pre-trained language models like BERT for even better embeddings.
