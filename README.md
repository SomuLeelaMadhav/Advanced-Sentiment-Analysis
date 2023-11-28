# Advanced-Sentiment-Analysis
Ensemble, DeepLearning ( Random Forest, Gradient Boosting, RNN, LSTM, etc. )

Overview
This project implements sentiment analysis using a various methods. The goal is to classify movie reviews as positive or negative based on their content.

# Sentiment Analysis with LSTM

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)


## Getting Started

### Prerequisites

- Python 3.x
- pip (Python package installer)
- Jupyter Notebook (for running and experimenting with the code)

### Installation

1. Clone the repository

### Prerequisites

pip (Python package installer)
Jupyter Notebook (for running and experimenting with the code)
Installation


Navigate to the project directory:


cd sentiment-analysis-lstm
Install dependencies:


pip install -r requirements.txt
Project Structure
data/: Contains the dataset used for training and testing.
notebooks/: Jupyter notebooks for data exploration and model development.
src/: Source code for the sentiment analysis model.
saved_models/: Directory to store saved models.
Usage
To perform sentiment analysis on new data, you can use the trained model. See the predict_sentiment function in the src/sentiment_analysis.py file.

from keras.models import load_model
from src.sentiment_analysis import predict_sentiment

# Load the trained LSTM model
model = load_model('saved_models/lstm_model_keras_format')

# Example usage:
text = "This movie was fantastic!"
sentiment = predict_sentiment(model, text)
print(f"The sentiment of the text is: {sentiment}")
Model Training
The LSTM model is trained using the train_lstm_model function in the src/model_training.py file. See the Jupyter notebooks in the notebooks/ directory for detailed experimentation and training.

Evaluation
The performance of the model can be evaluated using metrics such as accuracy, precision, recall, and F1-score. See the evaluate_model function in the src/evaluation.py file.

## Results
The LSTM model achieved an accuracy of 85% on the test dataset. For more details, refer to the Results section in the project report.
### NB_C: Accuracy: 0.89
### Random Forest Accuracy: 0.8998
### Gradient Boosting Accuracy: 0.938
### LSTM Accuracy: 0.996


Dependencies
Keras
TensorFlow
NumPy
Pandas
Matplotlib
Scikit-learn
Contributing
Feel free to contribute by opening issues or submitting pull requests.

License
This project is licensed under the MIT License.
