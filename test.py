import torch
import utils.Dataset as dataset
from model import BERT
from transformers import BertForSequenceClassification, BertTokenizer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

tokenizer = BertTokenizer.from_pretrained("./datasets/tokenizer")
model = BertForSequenceClassification.from_pretrained('./results/model')

tokenizer.save_pretrained('./datasets/tokenizer')

print(model)


# 应用模型进行情感分析
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    prediction = outputs.logits.argmax(-1).item()
    return "Positive" if prediction == 1 else "Negative"


print(predict_sentiment("I love this movie, that sounds great!"))
