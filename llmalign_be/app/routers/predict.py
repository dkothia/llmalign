# app/routers/predict.py
from fastapi import APIRouter, Form
from app.services.predictor import run_prediction
import os

router = APIRouter()

AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
if not AWS_BUCKET_NAME:
    raise ValueError("AWS_BUCKET_NAME is not defined in the environment variables.")

@router.post("/", description="Predict using a fine-tuned model")
def predict(
    run_id: str = Form(...),
    model_name: str = Form(...),
    text: str = Form(...)
):
    result = run_prediction(run_id, model_name, text)
    return result
