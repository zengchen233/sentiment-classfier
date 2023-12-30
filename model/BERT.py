# Load model directly
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import os


model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
trainer = Trainer
training_arguments = TrainingArguments


def get_tokenizer():
    if os.path.exists('./datasets/tokenizer'):
        tokenizer = BertTokenizer.from_pretrained('./datasets/tokenizer')
    else:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        tokenizer.save_pretrained('./datasets/tokenizer')
    return tokenizer


def get_model():
    return model


def get_trainer():
    return trainer


def get_training_arguments():
    return training_arguments


def tokenize_func(data):
    tokenizer = get_tokenizer()
    return tokenizer(data['text'], padding="max_length", truncation=True, max_length=512)
