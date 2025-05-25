import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import os

def upload_model_to_s3(model_path, bucket_name="my-fake-bucket", s3_key="models/fashion_cnn.h5"):
    # Mock S3 with localstack or assume S3 credentials are set
    try:
        s3 = boto3.client('s3', aws_access_key_id='fake_access_key', aws_secret_access_key='fake_secret_key', endpoint_url='http://localhost:4566')
        s3.upload_file(model_path, bucket_name, s3_key)
        print(f"Uploaded {model_path} to s3://{bucket_name}/{s3_key}")
    except FileNotFoundError:
        print("Model file not found!")
    except NoCredentialsError:
        print("Credentials not available for S3 upload!")
    except ClientError as e:
        print(f"ClientError: {e}")

if __name__ == "__main__":
    # Example usage
    os.makedirs('models', exist_ok=True)
    fake_model_path = 'models/fashion_cnn.h5'
    with open(fake_model_path, 'w') as f:
        f.write('fake model content')  # Simulate a model file
    upload_model_to_s3(fake_model_path)
