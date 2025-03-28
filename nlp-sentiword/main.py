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
import requests
import zipfile
import io
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('omw-1.4', quiet=True)  # Open Multilingual WordNet

DATASET_ID = "jp797498e/twitter-entity-sentiment-analysis"
TEST_SIZE = 0.2
RANDOM_SEED = 42
MAX_FEATURES = 30000
NGRAM_RANGE = (1, 3)  # Use unigrams, bigrams and trigrams
SENTIWORDNET_URL = "https://github.com/aesuli/SentiWordNet/raw/master/data/SentiWordNet_3.0.0.txt"


def download_sentiwordnet():
    """Download and load SentiWordNet."""
    print("\nDownloading SentiWordNet...")
    try:
        response = requests.get(SENTIWORDNET_URL)
        response.raise_for_status()
        
        # Create directory if it doesn't exist
        os.makedirs('resources', exist_ok=True)
        
        # Save the file
        with open('resources/SentiWordNet_3.0.0.txt', 'wb') as f:
            f.write(response.content)
        
        print("SentiWordNet downloaded successfully")
        return 'resources/SentiWordNet_3.0.0.txt'
    except Exception as e:
        print(f"Error downloading SentiWordNet: {e}")
        return None


def load_sentiwordnet(file_path):
    """Load and parse SentiWordNet lexicon.
    
    SentiWordNet assigns sentiment scores to WordNet synsets using the following format:
    POS  ID  PosScore  NegScore  SynsetTerms  Gloss
    
    Where:
    - POS: Part of speech (a=adjective, n=noun, v=verb, r=adverb)
    - ID: Synset ID
    - PosScore: Positive score [0,1]
    - NegScore: Negative score [0,1]
    - SynsetTerms: List of terms in the synset
    - Gloss: Meaning/definition (not used in this implementation)
    """
    if not file_path or not os.path.exists(file_path):
        print("SentiWordNet file not found")
        return {}
    
    print("Loading SentiWordNet lexicon...")
    sentiwordnet = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as swn_file:
            for line in swn_file:
                if line.startswith('#'):
                    continue
                
                parts = line.strip().split('\t')
                if len(parts) < 6:
                    continue
                
                pos, id_str, pos_score, neg_score = parts[0], parts[1], parts[2], parts[3]
                
                # Convert string scores to float
                pos_score = float(pos_score)
                neg_score = float(neg_score)
                
                # Get synset terms
                synset_terms = parts[4].split()
                terms = [t.split('#')[0] for t in synset_terms]
                
                # Create a dictionary entry for each term with its POS
                for term in terms:
                    term_pos = f"{term}#{pos}"
                    if term_pos not in sentiwordnet:
                        sentiwordnet[term_pos] = {'pos': [], 'neg': []}
                    
                    sentiwordnet[term_pos]['pos'].append(pos_score)
                    sentiwordnet[term_pos]['neg'].append(neg_score)
                    
                    # Also store the term without POS for fallback
                    if term not in sentiwordnet:
                        sentiwordnet[term] = {'pos': [], 'neg': []}
                    
                    sentiwordnet[term]['pos'].append(pos_score)
                    sentiwordnet[term]['neg'].append(neg_score)
        
        # Calculate average scores for each term
        for term in sentiwordnet:
            sentiwordnet[term]['pos'] = sum(sentiwordnet[term]['pos']) / len(sentiwordnet[term]['pos'])
            sentiwordnet[term]['neg'] = sum(sentiwordnet[term]['neg']) / len(sentiwordnet[term]['neg'])
            # Add objectivity score
            sentiwordnet[term]['obj'] = 1 - (sentiwordnet[term]['pos'] + sentiwordnet[term]['neg'])
        
        print(f"SentiWordNet loaded with {len(sentiwordnet)} terms")
        return sentiwordnet
    except Exception as e:
        print(f"Error loading SentiWordNet: {e}")
        return {}


def get_wordnet_pos(tag):
    """Map NLTK POS tag to WordNet POS tag.
    
    Returns:
        A character denoting the WordNet POS tag:
        - 'a' for adjective
        - 'n' for noun 
        - 'v' for verb
        - 'r' for adverb
    """
    tag = tag[0].upper()
    tag_dict = {
        'J': 'a',  # Adjective
        'N': 'n',  # Noun
        'V': 'v',  # Verb
        'R': 'r'   # Adverb
    }
    return tag_dict.get(tag, 'n')  # Default to noun if tag is not found


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


def clean_text(text, use_stemming=True, use_lemmatization=True):
    """Clean text using regex patterns and apply stemming and/or lemmatization."""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s!?.,]', '', text)
    text = re.sub(r'([!?])\1+', r'\1', text)
    
    tokens = word_tokenize(text)
    
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    
    processed_tokens = []
    for token in tokens:
        if use_stemming:
            token = stemmer.stem(token)
            
        if use_lemmatization:
            token = lemmatizer.lemmatize(token)
            
        processed_tokens.append(token)
    
    text = ' '.join(processed_tokens)
    text = ' '.join(text.split())
    
    return text


def apply_text_preprocessing(data, text_col, use_stemming=True, use_lemmatization=True):
    """Apply text preprocessing to the dataset."""
    print("\nApplying text preprocessing...")
    print(f"Using stemming: {use_stemming}")
    print(f"Using lemmatization: {use_lemmatization}")
    
    data['processed_text'] = data[text_col].apply(
        lambda x: clean_text(x, use_stemming=use_stemming, use_lemmatization=use_lemmatization)
    )
    print("Text preprocessing completed")
    return data


class SentiWordNetTransformer(BaseEstimator, TransformerMixin):
    """Transform text data using SentiWordNet sentiment scores.
    
    This transformer extracts sentiment features from text using SentiWordNet.
    It considers the part-of-speech of each word to get more accurate sentiment scores.
    """
    
    def __init__(self, sentiwordnet):
        self.sentiwordnet = sentiwordnet
        self.lemmatizer = WordNetLemmatizer()
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Calculate SentiWordNet features for each text.
        
        Features extracted:
        - Average positive score
        - Average negative score
        - Average objectivity score
        - Compound score (positive - negative)
        - Maximum positive score
        - Maximum negative score
        - Positive-negative ratio for emotional content
        - Weighted sentiment by term position
        """
        print("\nExtracting enhanced SentiWordNet features...")
        
        features = np.zeros((len(X), 8))  # 8 features
        
        for i, text in enumerate(X):
            tokens = word_tokenize(text)
            if not tokens:
                continue
                
            pos_tags = nltk.pos_tag(tokens)
            
            pos_scores = []
            neg_scores = []
            obj_scores = []
            max_pos = 0
            max_neg = 0
            
            # For positional weighting - words at beginning and end often carry more sentiment
            n_tokens = len(tokens)
            weighted_pos_score = 0
            weighted_neg_score = 0
            weights_sum = 0
            
            for j, (token, tag) in enumerate(pos_tags):
                # Get WordNet POS
                wn_pos = get_wordnet_pos(tag)
                
                # Lemmatize the token according to its POS
                lemma = self.lemmatizer.lemmatize(token.lower(), pos=wn_pos)
                
                # First try with specific POS tag
                term_pos = f"{lemma}#{wn_pos}"
                
                # Position weight - higher weight for words at beginning and end
                position_weight = 1.0
                if n_tokens > 5:  # Only for longer texts
                    if j < n_tokens // 3:  # Beginning
                        position_weight = 1.5
                    elif j > (2 * n_tokens) // 3:  # End
                        position_weight = 1.3
                
                weights_sum += position_weight
                
                # Look up sentiment scores with POS tag first, then fallback to without POS
                if term_pos in self.sentiwordnet:
                    sentiment = self.sentiwordnet[term_pos]
                    pos_scores.append(sentiment['pos'])
                    neg_scores.append(sentiment['neg'])
                    obj_scores.append(sentiment['obj'])
                    
                    weighted_pos_score += sentiment['pos'] * position_weight
                    weighted_neg_score += sentiment['neg'] * position_weight
                    
                    max_pos = max(max_pos, sentiment['pos'])
                    max_neg = max(max_neg, sentiment['neg'])
                elif lemma in self.sentiwordnet:
                    sentiment = self.sentiwordnet[lemma]
                    pos_scores.append(sentiment['pos'])
                    neg_scores.append(sentiment['neg'])
                    obj_scores.append(sentiment['obj'])
                    
                    weighted_pos_score += sentiment['pos'] * position_weight
                    weighted_neg_score += sentiment['neg'] * position_weight
                    
                    max_pos = max(max_pos, sentiment['pos'])
                    max_neg = max(max_neg, sentiment['neg'])
            
            if pos_scores and neg_scores:
                # Basic features
                features[i, 0] = sum(pos_scores) / len(pos_scores)  # Average positive score
                features[i, 1] = sum(neg_scores) / len(neg_scores)  # Average negative score
                features[i, 2] = sum(obj_scores) / len(obj_scores) if obj_scores else 0  # Average objectivity score
                features[i, 3] = features[i, 0] - features[i, 1]  # Compound score
                
                # Advanced features
                features[i, 4] = max_pos  # Maximum positive
                features[i, 5] = max_neg  # Maximum negative
                
                # Emotional content ratio (positive to negative ratio)
                pos_sum = sum(pos_scores)
                neg_sum = sum(neg_scores)
                features[i, 6] = pos_sum / (neg_sum + 0.001)  # Add small value to avoid division by zero
                
                # Weighted sentiment by position
                features[i, 7] = (weighted_pos_score - weighted_neg_score) / weights_sum if weights_sum > 0 else 0
        
        print(f"Enhanced SentiWordNet features extracted. Shape: {features.shape}")
        return features


def extract_features(X_train, X_test, sentiwordnet):
    """Extract TF-IDF and SentiWordNet features from text data."""
    
    print("\nExtracting features...")
    
    tfidf = TfidfVectorizer(
        stop_words='english',
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=2,
        max_df=0.95
    )
    
    swn_transformer = SentiWordNetTransformer(sentiwordnet)
    
    # Combine features using FeatureUnion
    features = FeatureUnion([
        ('tfidf', tfidf),
        ('sentiwordnet', swn_transformer)
    ])
    
    X_train_features = features.fit_transform(X_train)
    X_test_features = features.transform(X_test)
    
    print(f"Feature extraction complete. Number of features: {X_train_features.shape[1]}")
    return X_train_features, X_test_features, features


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
    
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('nlp_confusion_matrix.png')
    plt.close()
    
    # Export metrics to PNG files
    export_metrics_to_png(report, target_names, accuracy)
    
    return accuracy


def export_metrics_to_png(report, target_names, accuracy):
    """Export classification metrics to PNG files."""
    print("\nExporting metrics to PNG files...")
    
    # Overall accuracy plot
    plt.figure(figsize=(6, 4))
    plt.bar(['Accuracy'], [accuracy], color='blue')
    plt.ylim(0, 1.0)
    plt.title('Model Accuracy')
    plt.ylabel('Score')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('metrics_accuracy.png')
    plt.close()
    
    # Extract metrics for each class
    metrics = {'Precision': [], 'Recall': [], 'F1-Score': []}
    for label in target_names:
        if label in report:
            metrics['Precision'].append(report[label]['precision'])
            metrics['Recall'].append(report[label]['recall'])
            metrics['F1-Score'].append(report[label]['f1-score'])
    
    # Class-specific metrics
    plt.figure(figsize=(12, 8))
    x = np.arange(len(target_names))
    width = 0.25
    
    plt.bar(x - width, metrics['Precision'], width, label='Precision')
    plt.bar(x, metrics['Recall'], width, label='Recall')
    plt.bar(x + width, metrics['F1-Score'], width, label='F1-Score')
    
    plt.xlabel('Classes')
    plt.ylabel('Score')
    plt.title('Precision, Recall, and F1-Score by Class')
    plt.xticks(x, target_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('metrics_by_class.png')
    plt.close()
    
    # Macro and weighted averages
    plt.figure(figsize=(10, 6))
    avg_metrics = {
        'Macro Avg': [report['macro avg']['precision'], report['macro avg']['recall'], report['macro avg']['f1-score']],
        'Weighted Avg': [report['weighted avg']['precision'], report['weighted avg']['recall'], report['weighted avg']['f1-score']]
    }
    
    labels = ['Precision', 'Recall', 'F1-Score']
    x = np.arange(len(labels))
    width = 0.35
    
    plt.bar(x - width/2, avg_metrics['Macro Avg'], width, label='Macro Average')
    plt.bar(x + width/2, avg_metrics['Weighted Avg'], width, label='Weighted Average')
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Macro and Weighted Average Metrics')
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig('metrics_averages.png')
    plt.close()
    
    print("Metrics exported successfully!")


def analyze_feature_importance(model, feature_extractor):
    """Analyze and display most important features."""
    try:
        # Get feature names for TF-IDF (first part of FeatureUnion)
        tfidf_transformer = feature_extractor.transformer_list[0][1]
        feature_names = list(tfidf_transformer.get_feature_names_out())
        
        # Add SentiWordNet feature names
        feature_names.extend([
            'SWN_Positive', 'SWN_Negative', 'SWN_Objective', 'SWN_Compound', 
            'SWN_MaxPos', 'SWN_MaxNeg', 'SWN_PosNegRatio', 'SWN_PositionalWeight'
        ])
        
        coefficients = model.coef_[0]
        
        # Make sure we have the right number of feature names
        if len(feature_names) != len(coefficients):
            print(f"Feature names length ({len(feature_names)}) doesn't match coefficients length ({len(coefficients)})")
            feature_names = [f"Feature_{i}" for i in range(len(coefficients))]
        
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
        
    except (AttributeError, IndexError) as e:
        print(f"Could not analyze feature importance due to: {e}")


def map_labels_to_ids(labels):
    """Map text labels to numeric IDs."""
    unique_labels = sorted(list(set(labels)))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    label_ids = [label_map[label] for label in labels]
    return label_ids, label_map


def test_example_predictions(model, feature_extractor, label_map, sentiwordnet):
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
    
    example_features = feature_extractor.transform(processed_texts)
    
    predictions = model.predict(example_features)
    probabilities = model.predict_proba(example_features)
    
    id_to_label = {v: k for k, v in label_map.items()}
    
    print("\nExample Predictions:")
    
    # Extract SentiWordNet scores for each example
    swn_transformer = SentiWordNetTransformer(sentiwordnet)
    swn_scores = swn_transformer.transform(processed_texts)
    
    for i, (text, pred, proba) in enumerate(zip(example_texts, predictions, probabilities)):
        predicted_label = id_to_label[pred]
        confidence = max(proba)
        
        # Extract detailed SentiWordNet scores
        pos_score = swn_scores[i, 0]
        neg_score = swn_scores[i, 1]
        obj_score = swn_scores[i, 2]
        compound = swn_scores[i, 3]
        max_pos = swn_scores[i, 4]
        max_neg = swn_scores[i, 5]
        pos_neg_ratio = swn_scores[i, 6]
        positional_weight = swn_scores[i, 7]
        
        print(f"Text: '{text}'")
        print(f"Predicted Sentiment: {predicted_label} (Confidence: {confidence:.4f})")
        print(f"SentiWordNet Scores:")
        print(f"  - Basic: Positive: {pos_score:.4f}, Negative: {neg_score:.4f}, Objective: {obj_score:.4f}, Compound: {compound:.4f}")
        print(f"  - Advanced: Max Pos: {max_pos:.4f}, Max Neg: {max_neg:.4f}, Pos/Neg Ratio: {pos_neg_ratio:.4f}, Positional: {positional_weight:.4f}")
        
        proba_str = ", ".join([f"{id_to_label[i]}: {p:.4f}" for i, p in enumerate(proba)])
        print(f"Probabilities: {proba_str}\n")


def main():
    """Main function to orchestrate the NLP sentiment analysis workflow."""
    # Download and load SentiWordNet
    swn_file_path = download_sentiwordnet()
    sentiwordnet = load_sentiwordnet(swn_file_path)
    
    dataset_path = download_dataset()
    csv_file_path = find_csv_file(dataset_path)
    data = load_and_explore_data(csv_file_path)
    
    _, sentiment_col, text_col = identify_columns(data)
    visualize_sentiment_distribution(data, sentiment_col)
    
    clean_data = preprocess_data(data)
    
    processed_data = apply_text_preprocessing(
        clean_data, 
        text_col,
        use_stemming=False,
        use_lemmatization=True
    )
    
    y_labels = processed_data[sentiment_col].tolist()
    y_ids, label_map = map_labels_to_ids(y_labels)
    
    X = processed_data['processed_text']
    y = y_ids
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    
    X_train_features, X_test_features, feature_extractor = extract_features(X_train, X_test, sentiwordnet)
    
    model = train_model(X_train_features, y_train)
    
    evaluate_model(model, X_test_features, y_test, {v: k for k, v in label_map.items()})
    
    analyze_feature_importance(model, feature_extractor)
    
    test_example_predictions(model, feature_extractor, label_map, sentiwordnet)
    
    print("\nNLP Sentiment Analysis with SentiWordNet Complete!")


if __name__ == "__main__":
    main()
