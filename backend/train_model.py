import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

# 1. Load and preprocess data
df = pd.read_csv('../Training_Essay_Data.csv')
df = df.dropna(subset=['text', 'generated'])

# Encode labels (0 = Human, 1 = AI)
le = LabelEncoder()
df['generated'] = le.fit_transform(df['generated'])

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(), df['generated'].tolist(), test_size=0.2, random_state=42
)

# 2. Tokenization
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def tokenize(batch):
    return tokenizer(batch, padding='max_length', truncation=True, max_length=256, return_tensors='pt')

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=256)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=256)

class EssayDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = EssayDataset(train_encodings, train_labels)
val_dataset = EssayDataset(val_encodings, val_labels)

# 3. Load model
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

# 4. Training arguments
model_dir = './model'
training_args = TrainingArguments(
    output_dir=model_dir,
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    # evaluation_strategy='epoch',
    # save_strategy='epoch',
    logging_dir='./logs',
    logging_steps=10,
    save_total_limit=2,
)

# 5. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# 6. Train (auto-resume from latest checkpoint if available)
last_ckpt = get_last_checkpoint(model_dir)
if last_ckpt:
    print(f"Resuming training from checkpoint: {last_ckpt}")
    trainer.train(resume_from_checkpoint=last_ckpt)
else:
    trainer.train()

# 7. Save model and tokenizer
model.save_pretrained('./model')
tokenizer.save_pretrained('./model')

print("Training complete. Model saved to ./model")