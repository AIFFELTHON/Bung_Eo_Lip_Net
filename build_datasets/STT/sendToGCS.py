from google.cloud import storage
import os
#로컬에 있는 파일 GCS로 옮기기
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="key.json"

def upload_blob(bucket_name, source_file_name, destination_blob_name):

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename("./wavs/"+source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )

