# BERT Fine-tuning for Movie Review Sentiment Analysis

This project demonstrates how to fine-tune a BERT model for sentiment analysis on movie reviews using the Hugging Face Transformers library.

## 📋 Overview

The project implements a binary sentiment classification system that can predict whether a movie review is positive or negative. It uses the pre-trained BERT-base-uncased model and fine-tunes it on a sample of IMDB movie reviews.

## 🚀 Features

- **BERT Fine-tuning**: Uses `BertForSequenceClassification` for sentiment analysis
- **Data Preprocessing**: Handles text tokenization, padding, and truncation
- **Model Training**: Implements training with evaluation metrics
- **Model Persistence**: Saves and loads trained models
- **Prediction Pipeline**: Includes a complete inference function for new reviews

## 📁 Project Structure

```
Bert Fintune/
├── BERT_finetune_Movie_review_analysis.ipynb  # Main notebook
├── IMDB_Dataset_sample.xlsx                    # Sample dataset
└── README.md                                   # This file
```

## 🔧 Requirements

Install the required packages:

```bash
pip install transformers datasets torch scikit-learn pandas openpyxl
```

### Key Dependencies

- **transformers**: Hugging Face library for transformer models
- **datasets**: Library for handling datasets
- **torch**: PyTorch for deep learning
- **scikit-learn**: For train-test splitting
- **pandas**: For data manipulation
- **openpyxl**: For reading Excel files

## 📊 Dataset

The project uses a sample IMDB dataset (`IMDB_Dataset_sample.xlsx`) containing:
- **review**: Movie review text
- **sentiment**: Binary labels ("positive" or "negative")

## 🔄 Workflow

### 1. Data Loading and Preprocessing
- Load the Excel dataset using pandas
- Convert sentiment labels to numerical format (positive=1, negative=0)
- Split data into training and testing sets (80/20 split)

### 2. Tokenization
- Use `BertTokenizer` to tokenize text
- Apply padding (`max_length=128`) and truncation
- Convert to Hugging Face Dataset format

### 3. Model Setup
- Load pre-trained `BertForSequenceClassification` model
- Configure for binary classification (`num_labels=2`)

### 4. Training Configuration
```python
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=1000
)
```

### 5. Model Training and Evaluation
- Train the model using the `Trainer` class
- Evaluate performance on test set
- Save the trained model and tokenizer

### 6. Inference
- Load saved model and tokenizer
- Implement prediction function for new reviews
- Map predictions back to sentiment labels

## 🎯 Key Parameters

- **Model**: `bert-base-uncased`
- **Max Length**: 128 tokens
- **Batch Size**: 8 (training), 8 (evaluation)
- **Learning Rate**: 2e-5
- **Epochs**: 3
- **Weight Decay**: 0.01

## 💻 Usage

### Training the Model

Run the complete notebook `BERT_finetune_Movie_review_analysis.ipynb` to:
1. Load and preprocess the data
2. Train the BERT model
3. Evaluate performance
4. Save the trained model

### Making Predictions

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load saved model
model = BertForSequenceClassification.from_pretrained("./sentiment_model")
tokenizer = BertTokenizer.from_pretrained("./sentiment_model")

# Predict sentiment
review = "I really loved this movie. The story was amazing!"
sentiment = predict_sentiment(review, model, tokenizer)
print(f"Predicted Sentiment: {sentiment}")
```

## 📈 Model Performance

The model achieves strong performance on sentiment classification:
- **Evaluation Loss**: 0.463 (after 3 epochs)
- **Evaluation Runtime**: 1.58 seconds
- **Evaluation Speed**: 141.48 samples/second
- **Training Dataset Size**: 891 samples
- **Test Dataset Size**: 223 samples (20% split)
- **Model Architecture**: BERT-base-uncased (110M parameters)

## 🔍 Key Concepts Explained

### Tokenization Parameters
- **`padding="max_length"`**: Ensures uniform sequence length
- **`truncation=True`**: Truncates long sequences to max_length
- **`max_length=128`**: Sets fixed sequence size for efficiency

### Model Architecture
- **BERT-base-uncased**: 12-layer transformer model
- **Sequence Classification Head**: Added on top for sentiment prediction
- **Binary Classification**: Outputs logits for positive/negative sentiment

## 🛠️ Technical Notes

- The model expects input format with `input_ids`, `attention_mask`, and `labels`
- Sentiment column is renamed to `labels` for Hugging Face compatibility
- Model automatically handles GPU/CPU device placement
- Logits are converted to probabilities using softmax activation

## 📝 Future Improvements

- **Model Variants**: Experiment with BERT-large, RoBERTa, or DistilBERT for better performance
- **Data Augmentation**: Implement text augmentation techniques (back-translation, paraphrasing)
- **Evaluation Metrics**: Add accuracy, precision, recall, and F1-score calculations
- **Cross-Validation**: Implement k-fold cross-validation for robust evaluation
- **Hyperparameter Tuning**: Use grid search or Bayesian optimization
- **Production Features**: Add model versioning, API endpoints, and monitoring
- **Data Expansion**: Train on full IMDB dataset (50K reviews) for better generalization

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve this project.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Note**: This project is for educational purposes. Please ensure you have appropriate licenses for the datasets and models used. The BERT model used is subject to its own licensing terms from Hugging Face.
