# app/services/s3_utils.py
import boto3
import os
import tempfile

AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")

s3 = boto3.client("s3")

def upload_file_to_s3(filename: str, content: bytes) -> str:
    s3.put_object(
        Bucket=AWS_BUCKET_NAME,
        Key=filename,
        Body=content,
        ContentType="application/zip"
    )
    return f"s3://{AWS_BUCKET_NAME}/{filename}"


def download_model_from_s3(s3_key: str) -> str:
    local_path = tempfile.NamedTemporaryFile(delete=False).name
    s3.download_file(Bucket=AWS_BUCKET_NAME, Key=s3_key, Filename=local_path)
    return local_path



