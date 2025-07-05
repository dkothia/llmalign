import os
import uuid
import shutil
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from huggingface_hub import HfApi, HfFolder, create_repo, upload_folder
from app.services.s3_utils import download_model_from_s3
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from peft import get_peft_model, LoraConfig, TaskType
from app.services.s3_utils import upload_file_to_s3


def detect_columns(df: pd.DataFrame):
    text_candidates = ["text", "prompt", "question", "input", "content"]
    label_candidates = ["label", "target", "output", "class", "response"]

    text_col = next((col for col in df.columns if col.lower() in text_candidates), df.columns[0])
    label_col = next((col for col in df.columns if col.lower() in label_candidates), df.columns[-1])
    return text_col, label_col


def train_model(dataset_path, model_name, task_type, target_column=None):
    run_id = str(uuid.uuid4())[:8]
    output_dir = f"./models/{run_id}"
    os.makedirs(output_dir, exist_ok=True)

    # Download the dataset from S3
    local_csv = download_model_from_s3(dataset_path)
    print("Downloaded CSV to:", local_csv)

    # Load data
    df = pd.read_csv(local_csv, on_bad_lines="skip")  # deprecated in pandas 2.x
    text_col, label_col = detect_columns(df)
    if target_column:
        label_col = target_column

    # Encode labels
    le = LabelEncoder()
    df[label_col] = le.fit_transform(df[label_col])

    # Save label map
    label_map = {str(i): label for i, label in enumerate(le.classes_)}
    with open(f"{output_dir}/labels.json", "w") as f:
        json.dump(label_map, f)

    # Split
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df[label_col], random_state=42)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch[text_col], truncation=True, padding='max_length')

    # Datasets
    train_ds = Dataset.from_pandas(train_df).map(tokenize, batched=True)
    val_ds = Dataset.from_pandas(val_df).map(tokenize, batched=True)
    train_ds = train_ds.rename_column(label_col, "labels")
    val_ds = val_ds.rename_column(label_col, "labels")

    # Model + PEFT config
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(le.classes_))
    target_modules = get_lora_target_modules(model)
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        inference_mode=False,
        target_modules=target_modules
    )
    model = get_peft_model(model, peft_config)

    # Metrics
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = preds.argmax(axis=1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="weighted")
        }

    # Training
    training_args = TrainingArguments(
        output_dir="./results",
        #evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_dir="./logs"
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics
    )

    trainer.train()
    metrics = trainer.evaluate()

    # Save model
    model.save_pretrained(output_dir)

    # Hugging Face Hub Upload
    hf_username = "dishantkothia"  # Replace with your HF username or get dynamically
    repo_name = f"llmalign-{run_id}"
    full_repo = f"{hf_username}/{repo_name}"

    # Create repo if needed
    create_repo(repo_id=repo_name, private=True, exist_ok=True)

    # Push model directory
    upload_folder(
        folder_path=output_dir,
        repo_id=full_repo,
        commit_message="Upload LoRA fine-tuned model",
    )

    hf_model_url = f"https://huggingface.co/{full_repo}"


    # Zip + Upload
    zip_path = f"{output_dir}.zip"
    shutil.make_archive(output_dir, 'zip', output_dir)
    with open(zip_path, "rb") as f:
        s3_path = upload_file_to_s3(f"models/{run_id}.zip", f.read())

    for root, dirs, files in os.walk(output_dir):
        for file in files:
            local_path = os.path.join(root, file)
            rel_path = os.path.relpath(local_path, output_dir)
            s3_key = f"models/{run_id}/{rel_path}"
            with open(local_path, "rb") as f:
                upload_file_to_s3(s3_key, f.read())

    return {
        "status": "success",
        "run_id": run_id,
        "model_saved_to": s3_path,
        "huggingface_url": hf_model_url,
        "text_column": text_col,
        "label_column": label_col,
        "metrics": metrics,
        "labels": label_map
    }

def get_lora_target_modules(model):
    # Print all module names for debugging
    # print([name for name, _ in model.named_modules()])
    model_type = type(model).__name__.lower()
    if "distilbert" in model_type:
        return ["q_lin", "v_lin"]
    elif "bert" in model_type or "roberta" in model_type:
        return ["query", "value"]
    elif "llama" in model_type:
        return ["q_proj", "v_proj"]
    elif "gpt" in model_type:
        return ["c_attn"]
    else:
        # Fallback: try to find modules with 'query' or 'q_proj' etc.
        candidates = []
        for name, _ in model.named_modules():
            if any(x in name for x in ["query", "q_proj", "q_lin"]):
                candidates.append(name.split(".")[-1])
        return list(set(candidates)) or ["query", "value"]

