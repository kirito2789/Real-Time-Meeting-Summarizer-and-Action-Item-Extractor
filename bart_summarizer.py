import os
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments
)
import kagglehub
import torch
import numpy as np
from tqdm import tqdm
import evaluate
from sklearn.metrics import precision_score, recall_score, f1_score

# 1️⃣ Download Dataset
print("Downloading Kaggle dataset...")
path = kagglehub.dataset_download("paultimothymooney/medical-speech-transcription-and-intent")
print("Dataset downloaded at:", path)

# Locate the CSV file
csv_path = os.path.join(
    path,
    "medical speech transcription and intent",
    "Medical Speech, Transcription, and Intent",
    "overview-of-recordings.csv"
)

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV file not found: {csv_path}")

# 2️⃣ Load and Clean Data
df = pd.read_csv(csv_path)
print(f"CSV loaded. Total rows: {len(df)}")

df = df[df["prompt"].notna()]
print(f"Rows after filtering missing prompts: {len(df)}")

df = df[["prompt", "phrase"]].dropna()

# Split data
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)

dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "test": Dataset.from_pandas(test_df)
})
print(dataset)

# 3️⃣ Load Model and Tokenizer
model_name = "facebook/bart-base"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# 4️⃣ Preprocessing
def preprocess_function(examples):
    inputs = examples["prompt"]
    targets = examples["phrase"]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=64, truncation=True, padding="max_length").input_ids
    model_inputs["labels"] = labels
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# 5️⃣ Training Arguments
training_args = TrainingArguments(
    output_dir="./bart_results",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    logging_steps=100,
    save_strategy="no",
    evaluation_strategy="no",
    learning_rate=5e-5
)

# 6️⃣ Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# 7️⃣ Train Model
print("Training started...")
trainer.train()

# 8️⃣ Evaluation Section
print("\n✅ Using device for evaluation...")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
model.eval()

# Initialize evaluation metrics
rouge = evaluate.load("rouge")

references = []
predictions = []

print("\n🔍 Generating predictions for evaluation (sample of 200 examples)...")
for example in tqdm(test_df.sample(200, random_state=42).itertuples(), total=200):
    input_text = example.prompt
    reference = example.phrase

    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=128, truncation=True).to(device)
    with torch.no_grad():
        summary_ids = model.generate(
            input_ids=input_ids,
            max_length=64,
            num_beams=4,
            early_stopping=True
        )

    prediction = tokenizer.decode(summary_ids[0].cpu(), skip_special_tokens=True)

    predictions.append(prediction)
    references.append(reference)

# Compute ROUGE
rouge_results = rouge.compute(predictions=predictions, references=references)

# Compute precision, recall, F1 (token-level)
y_true, y_pred = [], []
for ref, pred in zip(references, predictions):
    ref_tokens = set(ref.lower().split())
    pred_tokens = set(pred.lower().split())
    all_tokens = list(ref_tokens.union(pred_tokens))
    y_true.append([1 if t in ref_tokens else 0 for t in all_tokens])
    y_pred.append([1 if t in pred_tokens else 0 for t in all_tokens])

y_true_flat = np.concatenate(y_true)
y_pred_flat = np.concatenate(y_pred)

precision = precision_score(y_true_flat, y_pred_flat)
recall = recall_score(y_true_flat, y_pred_flat)
f1 = f1_score(y_true_flat, y_pred_flat)

# 9️⃣ Print Results
print("\n📊 Evaluation Results:")
print(f"ROUGE-1: {rouge_results['rouge1']:.4f}")
print(f"ROUGE-2: {rouge_results['rouge2']:.4f}")
print(f"ROUGE-L: {rouge_results['rougeL']:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# 🔟 Display some qualitative samples
print("\n🧩 Sample Predictions:\n")
for i in range(3):
    print(f"🗣️ Prompt: {test_df.iloc[i]['prompt']}")
    print(f"📝 Generated: {predictions[i]}")
    print(f"🎯 Reference: {references[i]}")
    print("-" * 50)
