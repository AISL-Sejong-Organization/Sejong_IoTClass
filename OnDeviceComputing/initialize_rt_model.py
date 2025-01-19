import csv
import requests
import os
import json
import base64
from datetime import datetime
from uuid import uuid4
from PIL import Image
import numpy as np

# --------------------------------------------------------------------------------
# (1) Mobius 플랫폼 설정 (전역 상수 및 헤더)
# --------------------------------------------------------------------------------
MOBIUS_BASE_URL = "http://203.250.148.120:20519/Mobius/"
# MOBIUS_AE_NAME = "AIOTclass-OC"
MOBIUS_AE_NAME = "test-OC-resnet"
MOBIUS_SENSOR_DATA_CNT_NAME = "sensor_data"
MOBIUS_INFERENCING_RESULT_CNT_NAME = "inferencing_result"
MOBIUS_MODEL_REPOSITORY_CNT_NAME = "model_repository"

MOBIUS_BASE_AE_URL = os.path.join(MOBIUS_BASE_URL, MOBIUS_AE_NAME)

HEADERS_CNT = {
    'Accept': 'application/json',
    'X-M2M-RI': '12345',
    'X-M2M-Origin': 'SOrigin',
    'Content-Type': 'application/vnd.onem2m-res+json; ty=3'
}
HEADERS_CIN = {
    'Accept': 'application/json',
    'X-M2M-RI': '12345',
    'X-M2M-Origin': 'SOrigin',
    'Content-Type': 'application/vnd.onem2m-res+json; ty=4'
}


# --------------------------------------------------------------------------------
# (2) 모델 파일(Base64) 인코딩 함수
# --------------------------------------------------------------------------------
def encode_model(file_path):
    """
    .pth 모델 파일을 Base64로 인코딩하여 반환
    """
    with open(file_path, "rb") as f:
        encoded_bytes = base64.b64encode(f.read())
        encoded_str = encoded_bytes.decode('utf-8')  # JSON에서 사용할 수 있도록 문자열로 변환
    return encoded_str


# --------------------------------------------------------------------------------
# (3) 컨테이너 생성 및 데이터 업로드 함수
# --------------------------------------------------------------------------------
def create_container(digit, base_url):
    """
    - Mobius AE 하위에 컨테이너를 생성
    - digit(또는 이름)에 맞춰 컨테이너 생성 시 필요한 body 구성
    """
    container_name = f"{digit}"  # 예: "sensor_data", "inferencing_result", "model_repository"
    url = f"{base_url}"
    body = {
        "m2m:cnt": {
            "rn": container_name,
            "lbl": [digit],
            "mbs": 10000000  # 최대 크기 제한(mbs) 조정
        }
    }
    response = requests.post(url, headers=HEADERS_CNT, json=body)
    if response.status_code in [200, 201]:
        print(f"Container '{container_name}' created successfully at {base_url}.")
    else:
        print(f"Failed to create container '{container_name}' at {base_url}: {response.text}")


def upload_data(container_name, data, base_url):
    """
    - 특정 컨테이너(Mobius)로 data 업로드
    - data는 dict 형태
    """
    url = f"{base_url}/{container_name}"
    body = {
        "m2m:cin": {
            "con": data
        }
    }
    response = requests.post(url, headers=HEADERS_CIN, json=body)
    if response.status_code in [200, 201]:
        print(f"Data uploaded to '{container_name}' successfully at {base_url}.")
    else:
        print(f"Failed to upload data to '{container_name}' at {base_url}: {response.text}")


# --------------------------------------------------------------------------------
# (4) 컨테이너 생성 및 모델 업로드(메인 기능)
# --------------------------------------------------------------------------------
def create_rt_and_model_upload(model_path):
    """
    - sensor_data, inferencing_result, model_repository 컨테이너를 순차적으로 생성
    - 모델 파일(.pth)을 Base64로 인코딩해 model_repository 컨테이너에 업로드
    """
    # (4.1) 컨테이너 생성
    create_container(MOBIUS_SENSOR_DATA_CNT_NAME, MOBIUS_BASE_AE_URL)
    create_container(MOBIUS_INFERENCING_RESULT_CNT_NAME, MOBIUS_BASE_AE_URL)
    create_container(MOBIUS_MODEL_REPOSITORY_CNT_NAME, MOBIUS_BASE_AE_URL)

    # (4.2) 모델 업로드
    metadata = {
        "model_name": "CNN",
        "version": "1.0",
        "description": "This is a CNN model for camera sensor data classification."
    }
    encoded_model = encode_model(model_path)
    model_con = {
        "metadata": metadata,
        "model_file": encoded_model
    }
    upload_data(MOBIUS_MODEL_REPOSITORY_CNT_NAME, model_con, MOBIUS_BASE_AE_URL)


# --------------------------------------------------------------------------------
# (5) 메인 실행부
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    # 데이터 파일 경로
    MODEL_FILE_PATH = "./model/mnist_cnn.pth"
    create_rt_and_model_upload(MODEL_FILE_PATH)
