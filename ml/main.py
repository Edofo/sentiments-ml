"""Twitter Sentiment Analysis using Machine Learning.

This module implements sentiment analysis on Twitter data using classical ML approaches.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Constants
DATASET_ID = "jp797498e/twitter-entity-sentiment-analysis"
TEST_SIZE = 0.2
RANDOM_SEED = 42
MAX_FEATURES = 5000
LOGISTIC_MAX_ITER = 1000
RF_N_ESTIMATORS = 100


def download_dataset():
    """Download the Twitter sentiment dataset from Kaggle."""
    print("Downloading dataset...")
    path = kagglehub.dataset_download(DATASET_ID)
    print(f"Path to dataset files: {path}")
    return path


def find_csv_file(dataset_path):
    """Find the appropriate CSV file in the dataset directory.
    
    Prefers training file over validation file.
    """
    training_csv = None
    validation_csv = None

    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.csv'):
                if 'training' in file.lower():
                    training_csv = os.path.join(root, file)
                elif 'validation' in file.lower():
                    validation_csv = os.path.join(root, file)

    csv_file_path = training_csv if training_csv else validation_csv
    if not csv_file_path:
        raise FileNotFoundError("No CSV files found in the downloaded dataset.")
    
    print(f"\nLoading dataset from: {csv_file_path}")
    return csv_file_path


def load_and_explore_data(file_path):
    """Load the dataset and print exploratory information."""
    data = pd.read_csv(file_path)
    
    print("\nDataset Information:")
    print(f"Number of samples: {data.shape[0]}")
    print(f"Number of features: {data.shape[1]}")
    print("\nColumn names:", data.columns.tolist())
    print("\nFirst few rows:")
    print(data.head())
    
    return data


def identify_columns(data):
    """Identify the entity, sentiment, and text columns in the dataset."""
    columns = data.columns.tolist()
    
    entity_col = columns[1]  # Entity column (e.g., "Borderlands")
    sentiment_col = columns[2]  # Sentiment column (e.g., "Positive")
    text_col = columns[3]  # Text content column
    
    return entity_col, sentiment_col, text_col


def visualize_sentiment_distribution(data, sentiment_col):
    """Visualize the distribution of sentiment values."""
    print(f"\nSentiment Distribution (column '{sentiment_col}'):")
    sentiment_counts = data[sentiment_col].value_counts()
    print(sentiment_counts)
    
    plt.figure(figsize=(10, 6))
    sns.countplot(x=sentiment_col, data=data)
    plt.title('Sentiment Distribution')
    plt.savefig('sentiment_distribution.png')
    plt.close()


def preprocess_data(data):
    """Handle missing values in the dataset."""
    print("\nMissing values:")
    print(data.isnull().sum())
    
    return data.fillna("")


def extract_features(x_train, x_test):
    """Extract TF-IDF features from text data."""
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1, 2))
    x_train_features = vectorizer.fit_transform(x_train)
    x_test_features = vectorizer.transform(x_test)
    
    return x_train_features, x_test_features, vectorizer


def train_logistic_regression(x_train, y_train):
    """Train a Logistic Regression model."""
    print("\nTraining Logistic Regression model...")
    model = LogisticRegression(max_iter=LOGISTIC_MAX_ITER, random_state=RANDOM_SEED)
    model.fit(x_train, y_train)
    return model


def train_random_forest(x_train, y_train):
    """Train a Random Forest model."""
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_SEED)
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_test, y_test, model_name):
    """Evaluate model performance on test data."""
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"{model_name} Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report ({model_name}):")
    print(classification_report(y_test, y_pred))
    
    return accuracy, y_pred


def visualize_model_comparison(models, accuracies):
    """Visualize the performance comparison between models."""
    plt.figure(figsize=(10, 6))
    sns.barplot(x=models, y=accuracies)
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.savefig('model_comparison.png')
    plt.close()


def predict_sentiment(text, model, vectorizer):
    """Predict sentiment of new text using the trained model."""
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)[0]
    return prediction


def test_example_predictions(model, vectorizer):
    """Test the model on example texts."""
    example_texts = [
        "I love this product, it's amazing!",
        "This is the worst experience I've ever had.",
        "The service was okay, nothing special."
    ]

    print("\nExample Predictions:")
    for text in example_texts:
        sentiment = predict_sentiment(text, model, vectorizer)
        print(f"Text: '{text}'")
        print(f"Predicted Sentiment: {sentiment}\n")


def main():
    """Main function to orchestrate the sentiment analysis workflow."""
    # Download and load data
    dataset_path = download_dataset()
    csv_file_path = find_csv_file(dataset_path)
    data = load_and_explore_data(csv_file_path)
    
    # Identify columns and explore data
    _, sentiment_col, text_col = identify_columns(data)
    visualize_sentiment_distribution(data, sentiment_col)
    
    # Preprocess data
    clean_data = preprocess_data(data)
    
    # Prepare features and split data
    x = clean_data[text_col]
    y = clean_data[sentiment_col]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    print(f"\nTraining set size: {x_train.shape[0]}")
    print(f"Testing set size: {x_test.shape[0]}")
    
    # Extract features
    x_train_tfidf, x_test_tfidf, vectorizer = extract_features(x_train, x_test)
    
    # Train and evaluate models
    lr_model = train_logistic_regression(x_train_tfidf, y_train)
    accuracy_lr, _ = evaluate_model(lr_model, x_test_tfidf, y_test, "Logistic Regression")
    
    rf_model = train_random_forest(x_train_tfidf, y_train)
    accuracy_rf, _ = evaluate_model(rf_model, x_test_tfidf, y_test, "Random Forest")
    
    # Compare models
    model_names = ['Logistic Regression', 'Random Forest']
    accuracies = [accuracy_lr, accuracy_rf]
    visualize_model_comparison(model_names, accuracies)
    
    # Select the best model (highest accuracy)
    best_model = lr_model if accuracy_lr >= accuracy_rf else rf_model
    
    # Test predictions
    test_example_predictions(best_model, vectorizer)
    
    print("Sentiment Analysis Model Training Complete!")


if __name__ == "__main__":
    main()
