import os
import zipfile
import tempfile
import torch
import json
import shutil
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from app.services.s3_utils import download_model_from_s3


def run_prediction(run_id, model_name, text):
    # Step 1: Download model zip to a temp file
    with tempfile.NamedTemporaryFile(suffix=".zip") as tmp_zip:
        try:
            s3_zip_path = download_model_from_s3(f"models/{run_id}.zip")
            with open(s3_zip_path, "rb") as src, open(tmp_zip.name, "wb") as dst:
                dst.write(src.read())
        except Exception as e:
            raise RuntimeError(f"Failed to download model: {e}")

        # Step 2: Extract to a temp directory
        extract_dir = tempfile.mkdtemp()
        try:
            with zipfile.ZipFile(tmp_zip.name, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

            # Step 3: Load tokenizer and model with LoRA adapter
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            base_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            model = PeftModel.from_pretrained(base_model, extract_dir)
            model.eval()

            # Step 4: Load label map (labels.json)
            label_map_path = os.path.join(extract_dir, "labels.json")
            if os.path.exists(label_map_path):
                with open(label_map_path, "r") as f:
                    label_map = json.load(f)
            else:
                label_map = {}

            # Step 5: Tokenize input and run prediction
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                pred_id = torch.argmax(outputs.logits, dim=1).item()
                pred_label = label_map.get(str(pred_id), str(pred_id))

            return {
                "input": text,
                "prediction": pred_label,
                "prediction_id": pred_id,
                "run_id": run_id
            }
        finally:
            # Always clean up the temp directory
            shutil.rmtree(extract_dir)
