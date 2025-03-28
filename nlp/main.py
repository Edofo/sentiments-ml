"""Twitter Sentiment Analysis using NLP techniques.

This module implements sentiment analysis on Twitter data using efficient NLP approaches.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

DATASET_ID = "jp797498e/twitter-entity-sentiment-analysis"
TEST_SIZE = 0.2
RANDOM_SEED = 42
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 3)  # Use unigrams, bigrams and trigrams


def download_dataset():
    """Download the Twitter sentiment dataset from Kaggle."""
    print("Downloading dataset...")
    path = kagglehub.dataset_download(DATASET_ID)
    print(f"Path to dataset files: {path}")
    return path


def find_csv_file(dataset_path):
    """Find the appropriate CSV file in the dataset directory."""
    training_csv = None
    validation_csv = None
    any_csv = None

    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.csv'):
                if any_csv is None:
                    any_csv = os.path.join(root, file)
                if 'training' in file.lower():
                    training_csv = os.path.join(root, file)
                elif 'validation' in file.lower():
                    validation_csv = os.path.join(root, file)

    csv_file_path = training_csv if training_csv else (validation_csv if validation_csv else any_csv)
    
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
    
    if len(columns) < 4:
        raise ValueError(f"Expected at least 4 columns, got {len(columns)}: {columns}")
    
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
    plt.savefig('nlp_sentiment_distribution.png')
    plt.close()


def preprocess_data(data):
    """Handle missing values in the dataset."""
    print("\nMissing values:")
    print(data.isnull().sum())
    
    return data.fillna("")


def clean_text(text):
    """Clean text using regex patterns."""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s!?.,]', '', text)
    text = re.sub(r'([!?])\1+', r'\1', text)
    text = ' '.join(text.split())
    return text


def apply_text_preprocessing(data, text_col):
    """Apply text preprocessing to the dataset."""
    print("\nApplying text preprocessing...")
    data['processed_text'] = data[text_col].apply(clean_text)
    print("Text preprocessing completed")
    return data


def extract_features(X_train, X_test):
    """Extract TF-IDF features from text data."""
    
    print("\nExtracting TF-IDF features...")
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=2,
        max_df=0.95
    )
    
    X_train_features = vectorizer.fit_transform(X_train)
    X_test_features = vectorizer.transform(X_test)
    
    print(f"Feature extraction complete. Number of features: {X_train_features.shape[1]}")
    return X_train_features, X_test_features, vectorizer


def train_model(X_train, y_train):
    """Train a LogisticRegression model with hyperparameter tuning."""
    print("\nTraining Logistic Regression model with GridSearchCV...")
    
    n_samples = X_train.shape[0]
    n_splits = min(3, n_samples // 10)  # Reduced from 5 to 3 for speed
    n_splits = max(2, n_splits)  # At least 2 folds
    
    print(f"Using {n_splits}-fold cross-validation")
    
    param_grid = {
        'C': [1.0, 10.0],  # Reduced parameter options
        'class_weight': ['balanced'],
        'solver': ['liblinear'],  # liblinear is faster for small datasets
        'max_iter': [1000]
    }
    
    grid_search = GridSearchCV(
        LogisticRegression(),
        param_grid,
        cv=n_splits,
        scoring='f1_weighted',  # Changed from 'f1' to 'f1_weighted' for multiclass
        n_jobs=1  # Set to 1 to avoid potential multiprocessing issues
    )
    
    grid_search.fit(X_train, y_train)
    
    print("\nBest parameters found:")
    print(grid_search.best_params_)
    
    return grid_search.best_estimator_


def evaluate_model(model, X_test, y_test, label_map):
    """Evaluate model performance on test data."""
    print("\nEvaluating model...")
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    
    target_names = [str(label_map[i]) for i in sorted(label_map.keys())]
    
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('nlp_confusion_matrix.png')
    plt.close()
    
    return accuracy


def analyze_feature_importance(model, vectorizer):
    """Analyze and display most important features."""
    try:
        feature_names = vectorizer.get_feature_names_out()
        coefficients = model.coef_[0]
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': abs(coefficients)
        })
        
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        print("\nTop 20 most important features:")
        print(feature_importance.head(20))
        
        plt.figure(figsize=(12, 8))
        top_features = feature_importance.head(20)
        sns.barplot(x='importance', y='feature', data=top_features)
        plt.title('Top 20 Most Important Features')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        
    except (AttributeError, IndexError):
        print("Could not analyze feature importance due to model structure.")


def map_labels_to_ids(labels):
    """Map text labels to numeric IDs."""
    unique_labels = sorted(list(set(labels)))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    label_ids = [label_map[label] for label in labels]
    return label_ids, label_map


def test_example_predictions(model, vectorizer, label_map):
    """Test the model on example texts."""
    example_texts = [
        "I love this product, it's amazing!",
        "This is the worst experience I've ever had.",
        "The service was okay, nothing special.",
        "I don't know what to think about this.",
        "This game is absolutely fantastic, highly recommended!",
        "Terrible customer service and product quality.",
        "It's not bad, but I expected more for the price."
    ]

    processed_texts = [clean_text(text) for text in example_texts]
    
    example_features = vectorizer.transform(processed_texts)
    
    predictions = model.predict(example_features)
    probabilities = model.predict_proba(example_features)
    
    id_to_label = {v: k for k, v in label_map.items()}
    
    print("\nExample Predictions:")
    for text, pred, proba in zip(example_texts, predictions, probabilities):
        predicted_label = id_to_label[pred]
        confidence = max(proba)
        print(f"Text: '{text}'")
        print(f"Predicted Sentiment: {predicted_label} (Confidence: {confidence:.4f})")
        
        proba_str = ", ".join([f"{id_to_label[i]}: {p:.4f}" for i, p in enumerate(proba)])
        print(f"Probabilities: {proba_str}\n")


def main():
    """Main function to orchestrate the NLP sentiment analysis workflow."""
    dataset_path = download_dataset()
    csv_file_path = find_csv_file(dataset_path)
    data = load_and_explore_data(csv_file_path)
    
    _, sentiment_col, text_col = identify_columns(data)
    visualize_sentiment_distribution(data, sentiment_col)
    
    clean_data = preprocess_data(data)
    
    processed_data = apply_text_preprocessing(clean_data, text_col)
    
    y_labels = processed_data[sentiment_col].tolist()
    y_ids, label_map = map_labels_to_ids(y_labels)
    
    X = processed_data['processed_text']
    y = y_ids
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    
    X_train_tfidf, X_test_tfidf, vectorizer = extract_features(X_train, X_test)
    
    model = train_model(X_train_tfidf, y_train)
    
    evaluate_model(model, X_test_tfidf, y_test, {v: k for k, v in label_map.items()})
    
    analyze_feature_importance(model, vectorizer)
    
    test_example_predictions(model, vectorizer, label_map)
    
    print("\nNLP Sentiment Analysis Complete!")


if __name__ == "__main__":
    main()
