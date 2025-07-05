import boto3
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path="/Users/dishantkothia/llmalign/llmalign_be/.env")

def test_s3_connection():
    bucket_name = os.getenv("AWS_BUCKET_NAME")
    region = os.getenv("AWS_REGION")

    if not bucket_name or not region:
        raise ValueError("AWS_BUCKET_NAME or AWS_REGION is not defined in the environment variables.")

    s3 = boto3.client("s3", region_name=region)
    try:
        response = s3.list_objects_v2(Bucket=bucket_name)
        print("S3 Bucket Access Successful:", response)
    except Exception as e:
        print("S3 Bucket Access Failed:", str(e))

if __name__ == "__main__":
    test_s3_connection()