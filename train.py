import torch
import utils.Dataset as dataset
from model import BERT

SEED = 1129
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset = dataset.get_dataset()
tokenizer = BERT.get_tokenizer()
model = BERT.get_model()
model.to(device)

tokenized_datasets = dataset.map(BERT.tokenize_func, batched=True)  # 将各个text数据都进行tokenizer
tokenized_datasets = tokenized_datasets.remove_columns(['text'])  # 删除掉原始的text，保留经过处理之后的数据
tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')
tokenized_datasets.set_format('torch')

Trainer = BERT.get_trainer()
TrainingArguments = BERT.get_training_arguments()

training_args = TrainingArguments(
    output_dir='./results/trainer',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
    save_strategy='epoch',
    evaluation_strategy='epoch'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test']
)

trainer.train()  # 开始训练
trainer.evaluate()  # 评估模型
# trainer.save_model('./results/model')


# 应用模型进行情感分析
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    prediction = outputs.logits.argmax(-1).item()
    return "Positive" if prediction == 1 else "Negative"


print(predict_sentiment("I love u."))
