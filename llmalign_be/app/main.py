# app/main.py
from fastapi import FastAPI
from fastapi import Form
from transformers import AutoTokenizer
from peft import PeftModel
import torch
import logging
import os
from dotenv import load_dotenv

from app.routers import upload, train, predict
from app.services.predictor import run_prediction

# Load environment variables from .env file
load_dotenv(dotenv_path="/Users/dishantkothia/llmalign/llmalign_be/.env")

AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
AWS_REGION = os.getenv("AWS_REGION")

print("AWS_BUCKET_NAME:", AWS_BUCKET_NAME)
print("AWS_REGION:", AWS_REGION)

if not AWS_BUCKET_NAME or not AWS_REGION:
    raise ValueError("AWS_BUCKET_NAME or AWS_REGION is not defined in the environment variables.")

app = FastAPI(
    title="LLMAlign API",
    description="Fine-tune and deploy LLMs securely",
    version="0.1"
)

app.include_router(predict.router, prefix="/predict", tags=["Predict"])
app.include_router(upload.router, prefix="/upload", tags=["Upload"])
app.include_router(train.router, prefix="/train", tags=["Train"])

logging.basicConfig(level=logging.INFO)

@app.get("/")
def read_root():
    return {"message": "Welcome to LLMAlign backend!"}



