# %%
!pip install datasets -q

# %%
# Imports
import os
import json
import numpy as np
import pandas as pd
import re
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)


# %%
# Load and preprocess CUAD dataset
with open("/kaggle/input/dataset-cuad/CUADv1.json", "r") as f:
    data = json.load(f)

clauses = []
for item in data["data"]:
    context = item["paragraphs"][0]["context"]
    for qa in item["paragraphs"][0]["qas"]:
        if not qa["answers"]:
            continue
        match = re.search(r"related to (.+)", qa["question"])
        if match:
            clause_type = match.group(1).strip().strip("?")
        else:
            continue
        start = qa["answers"][0]["answer_start"]
        end = start + len(qa["answers"][0]["text"])
        extracted_text = context[start:end]
        clauses.append({
            "clause_text": extracted_text.strip(),
            "clause_type": clause_type
        })

df = pd.DataFrame(clauses)

# %%
def extract_core_label(label):
    match = re.search(r'"(.*?)"', label)
    return match.group(1) if match else label.strip()
    df['clause_type'] = df['clause_type'].apply(extract_core_label)


# %%
# Encode labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['clause_type'])

# Split into train and test
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['clause_text'].tolist(),
    df['label'].tolist(),
    test_size=0.2,
    stratify=df['label'],
    random_state=42
)


# %%
# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=256)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=256)


# %%
# Dataset wrapper
class ClauseDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = ClauseDataset(train_encodings, train_labels)
test_dataset = ClauseDataset(test_encodings, test_labels)


# %%
# Metric function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average='weighted')
    }


# %%
df.head()

# %%
df.shape

# %%
# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none"  # Disables wandb or other reporters
)


# %%
# Model
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=len(label_encoder.classes_)
)


# %%
# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)


# %%
# Train and evaluate
trainer.train()
trainer.evaluate()

# %%
# Save model + tokenizer
model.save_pretrained("/kaggle/working/barrister-bert")
tokenizer.save_pretrained("/kaggle/working/barrister-bert")

# %%
# Convert predictions to label indices
predictions = trainer.predict(test_dataset)
y_true = predictions.label_ids
y_pred = predictions.predictions.argmax(axis=1)

# Classification report
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))


# %%

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()



