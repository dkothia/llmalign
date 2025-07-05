# app/routers/upload.py
from fastapi import APIRouter, File, UploadFile
from app.services.s3_utils import upload_file_to_s3
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path="/Users/dishantkothia/llmalign/llmalign_be/.env")

router = APIRouter()

AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
if not AWS_BUCKET_NAME:
    raise ValueError("AWS_BUCKET_NAME is not defined in the environment variables.")

@router.post("/", description="Upload a file to S3")
async def upload_file(file: UploadFile = File(...)):
    print("Received file:", file.filename)
    #file_path = f"/tmp/{file.filename}"
    #print("Saving file to:", file_path)

    # Save file locally
    contents = await file.read()

    print("Uploading file to S3 bucket:", AWS_BUCKET_NAME)
    s3_key = f"datasets/{file.filename}"  # organized inside S3
    upload_file_to_s3(s3_key, contents)
    print("File uploaded successfully, S3 key:", s3_key)
    return {
        "message": f"File uploaded successfully to {AWS_BUCKET_NAME}",
        "s3_key": s3_key
    }
