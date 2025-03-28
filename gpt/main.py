import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
from transformers import pipeline
from torch.utils.data import Dataset, DataLoader

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

DATASET_ID = "jp797498e/twitter-entity-sentiment-analysis"
TEST_SIZE = 0.2
RANDOM_SEED = 42
MAX_LENGTH = 128
MODEL_NAME = "distilgpt2"
BATCH_SIZE = 4
LEARNING_RATE = 2e-5
NUM_EPOCHS = 8
SAMPLE_SIZE = 1000
GPT_MAX_LENGTH = 128

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=GPT_MAX_LENGTH):
        self.texts = texts.tolist() if hasattr(texts, 'tolist') else list(texts)
        self.labels = labels if isinstance(labels, list) else list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        if not text or text.isspace():
            text = "empty text"
            
        try:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label)
            }
        except Exception as e:
            print(f"Error tokenizing text: {text[:50]}... Error: {e}")
            input_ids = torch.zeros(self.max_length, dtype=torch.long)
            attention_mask = torch.zeros(self.max_length, dtype=torch.long)
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': torch.tensor(label)
            }


def download_dataset():
    print("Downloading dataset...")
    path = kagglehub.dataset_download(DATASET_ID)
    print(f"Path to dataset files: {path}")
    return path


def find_csv_file(dataset_path):
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
    data = pd.read_csv(file_path)
    
    print("\nDataset Information:")
    print(f"Number of samples: {data.shape[0]}")
    print(f"Number of features: {data.shape[1]}")
    print("\nColumn names:", data.columns.tolist())
    print("\nFirst few rows:")
    print(data.head())
    
    return data


def identify_columns(data):
    columns = data.columns.tolist()
    
    if len(columns) < 4:
        raise ValueError(f"Expected at least 4 columns, got {len(columns)}: {columns}")
    
    entity_col = columns[1]
    sentiment_col = columns[2]
    text_col = columns[3]
    
    return entity_col, sentiment_col, text_col


def visualize_sentiment_distribution(data, sentiment_col):
    print(f"\nSentiment Distribution (column '{sentiment_col}'):")
    sentiment_counts = data[sentiment_col].value_counts()
    print(sentiment_counts)
    
    plt.figure(figsize=(10, 6))
    sns.countplot(x=sentiment_col, data=data)
    plt.title('Sentiment Distribution')
    plt.savefig('nlp_sentiment_distribution.png')
    plt.close()


def preprocess_data(data):
    print("\nMissing values:")
    print(data.isnull().sum())
    
    return data.fillna("")


def clean_text(text, use_stemming=True, use_lemmatization=True):
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
    print("\nApplying text preprocessing...")
    print(f"Using stemming: {use_stemming}")
    print(f"Using lemmatization: {use_lemmatization}")
    
    data['processed_text'] = data[text_col].apply(
        lambda x: clean_text(x, use_stemming=use_stemming, use_lemmatization=use_lemmatization)
    )
    print("Text preprocessing completed")
    return data


def prepare_gpt_model(num_labels):
    print(f"\nLoading GPT model: {MODEL_NAME}")
    
    os.environ["PYTORCH_MPS_DEVICE_ENABLE"] = "0"
    device = torch.device("cpu")
    print(f"Using device: {device} (MPS disabled)")
    
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
        model = GPT2ForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id
            
        print("Successfully loaded GPT2 model for sequence classification")
    except Exception as e:
        print(f"Error loading GPT2 model: {e}")
        print("Trying with a general AutoModel approach...")
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id
    
    model = model.to(device)
    return model, tokenizer, device


def train_gpt_model(model, tokenizer, X_train, y_train, X_val, y_val, label_map, device):
    print("\nPreparing datasets for GPT model...")
    
    train_dataset = SentimentDataset(X_train, y_train, tokenizer)
    val_dataset = SentimentDataset(X_val, y_val, tokenizer)
    
    training_args = TrainingArguments(
        output_dir="./gpt_sentiment_model",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=20,
        eval_strategy="no",
        save_strategy="no",
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=2,
        disable_tqdm=False,
        save_total_limit=1,
    )
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels, predictions)}
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    print("\nTraining GPT model...")
    trainer.train()
    
    eval_results = trainer.evaluate()
    print(f"\nEvaluation results: {eval_results}")
    
    return model, trainer


def evaluate_gpt_model(model, tokenizer, X_test, y_test, label_map, device):
    print("\nEvaluating GPT model...")
    
    model.eval()
    model = model.to("cpu")
    
    test_dataset = SentimentDataset(X_test, y_test, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    all_predictions = []
    all_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to("cpu")
            attention_mask = batch['attention_mask'].to("cpu")
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            probs = torch.nn.functional.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    id_to_label = {v: k for k, v in label_map.items()}
    
    accuracy = accuracy_score(y_test, all_predictions)
    
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    
    target_names = [str(id_to_label[i]) for i in sorted(id_to_label.keys())]
    
    report = classification_report(y_test, all_predictions, target_names=target_names, output_dict=True)
    print(classification_report(y_test, all_predictions, target_names=target_names))
    
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, all_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix (GPT Model)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('gpt_confusion_matrix.png')
    plt.close()
    
    export_metrics_to_png(report, target_names, accuracy)
    
    return accuracy, all_predictions, all_probs


def test_gpt_example_predictions(model, tokenizer, label_map, device):
    example_texts = [
        "I love this product, it's amazing!",
        "This is the worst experience I've ever had.",
        "The service was okay, nothing special.",
        "I don't know what to think about this.",
        "This game is absolutely fantastic, highly recommended!",
        "Terrible customer service and product quality.",
        "It's not bad, but I expected more for the price."
    ]
    
    print("\nGPT Model Example Predictions:")
    
    model.eval()
    id_to_label = {v: k for k, v in label_map.items()}
    
    for text in example_texts:
        with torch.no_grad():
            inputs = tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=GPT_MAX_LENGTH,
                return_tensors='pt'
            ).to(device)
            
            outputs = model(**inputs)
            logits = outputs.logits
            
            probs = torch.nn.functional.softmax(logits[0], dim=0)
            pred = torch.argmax(probs).item()
            confidence = probs[pred].item()
            
            predicted_label = id_to_label.get(pred, f"Unknown-{pred}")
            
            print(f"Text: '{text}'")
            print(f"Predicted Sentiment: {predicted_label} (Confidence: {confidence:.4f})\n")


def export_metrics_to_png(report, target_names, accuracy):
    print("\nExporting metrics to PNG files...")
    
    plt.figure(figsize=(6, 4))
    plt.bar(['Accuracy'], [accuracy], color='blue')
    plt.ylim(0, 1.0)
    plt.title('Model Accuracy')
    plt.ylabel('Score')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('metrics_accuracy.png')
    plt.close()
    
    metrics = {'Precision': [], 'Recall': [], 'F1-Score': []}
    for label in target_names:
        if label in report:
            metrics['Precision'].append(report[label]['precision'])
            metrics['Recall'].append(report[label]['recall'])
            metrics['F1-Score'].append(report[label]['f1-score'])
    
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


def map_labels_to_ids(labels):
    unique_labels = sorted(list(set(labels)))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    label_ids = [label_map[label] for label in labels]
    return label_ids, label_map


def main():
    dataset_path = download_dataset()
    csv_file_path = find_csv_file(dataset_path)
    data = load_and_explore_data(csv_file_path)
    
    _, sentiment_col, text_col = identify_columns(data)
    visualize_sentiment_distribution(data, sentiment_col)
    
    if len(data) > SAMPLE_SIZE:
        print(f"\nLimiting dataset to {SAMPLE_SIZE} samples for faster training")
        data = data.sample(SAMPLE_SIZE, random_state=RANDOM_SEED)
    
    clean_data = preprocess_data(data)
    
    clean_data['processed_text'] = clean_data[text_col].str.lower()
    print("Simplified text preprocessing completed")
    
    y_labels = clean_data[sentiment_col].tolist()
    y_ids, label_map = map_labels_to_ids(y_labels)
    
    X = clean_data['processed_text']
    y = y_ids
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    
    num_labels = len(set(y))
    print(f"Number of sentiment classes: {num_labels}")
    
    model, tokenizer, device = prepare_gpt_model(num_labels)
    
    model, trainer = train_gpt_model(model, tokenizer, X_train, y_train, X_test, y_test, label_map, device)
    
    accuracy, predictions, probabilities = evaluate_gpt_model(model, tokenizer, X_test, y_test, label_map, device)
    
    test_gpt_example_predictions(model, tokenizer, label_map, device)
    
    model_save_path = "./gpt_sentiment_model"
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"\nSaved fine-tuned GPT model to {model_save_path}")
    
    print("\nGPT-based NLP Sentiment Analysis Complete!")


if __name__ == "__main__":
    main()
