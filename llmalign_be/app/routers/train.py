# app/routers/train.py
from fastapi import APIRouter, Form
from app.services.trainer import train_model
import os

router = APIRouter()

AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
if not AWS_BUCKET_NAME:
    raise ValueError("AWS_BUCKET_NAME is not defined in the environment variables.")

@router.post("/", description="Train a model using uploaded data")
def train(
    dataset_path: str = Form(...),
    model_name: str = Form(...),
    task_type: str = Form(...),
    target_column: str = Form(None)
):
    train_model(dataset_path, model_name, task_type, target_column)
    return {"message": f"Model {model_name} trained successfully and uploaded to {AWS_BUCKET_NAME}."}
