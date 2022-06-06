from google.cloud import storage
import os
#GCS에 저장된 파일 다운로드하기
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="key.json"

bucket_name = 'example_baker'    # 서비스 계정 생성한 bucket 이름 입력
source_blob_name = '/One/'    # GCP에 저장되어 있는 파일 명
destination_file_name = ''    # 다운받을 파일을 저장할 경로("local/path/to/file")

storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob(source_blob_name)

blob.download_to_filename(destination_file_name)