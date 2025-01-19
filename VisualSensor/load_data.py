import csv
import requests
import os
import json
from datetime import datetime
from uuid import uuid4
from PIL import Image
import numpy as np

# --------------------------------------------------------------------------------
# (1) Mobius 플랫폼 설정
# --------------------------------------------------------------------------------
MOBIUS_BASE_URL = "http://203.250.148.120:20519/Mobius/"

# (a) Trainset / (b) Testset 두 개의 AE
MOBIUS_AE_NAME     = "testman14"
MOBIUS_TEST_NAME   = "testman15"

MOBIUS_BASE_AE_URL = os.path.join(MOBIUS_BASE_URL, MOBIUS_AE_NAME)
MOBIUS_TEST_AE_URL = os.path.join(MOBIUS_BASE_URL, MOBIUS_TEST_NAME)

# 컨테이너(ty=3) 생성 시 필요한 헤더
HEADERS_CNT = {
    'Accept':       'application/json',
    'X-M2M-RI':     '12345',
    'X-M2M-Origin': 'SOrigin',
    'Content-Type': 'application/vnd.onem2m-res+json; ty=3'
}

# 콘텐츠 인스턴스(ty=4) 생성 시 필요한 헤더
HEADERS_CIN = {
    'Accept':       'application/json',
    'X-M2M-RI':     '12345',
    'X-M2M-Origin': 'SOrigin',
    'Content-Type': 'application/vnd.onem2m-res+json; ty=4'
}

# --------------------------------------------------------------------------------
# (2) Mobius 컨테이너 생성 함수
# --------------------------------------------------------------------------------
def create_container(digit: str, base_url: str) -> None:
    """
    base_url(AE) 아래, digit을 이름으로 하는 컨테이너 생성.
    예: digit="0" => container "0"
    """
    container_name = digit
    url = base_url
    body = {
        "m2m:cnt": {
            "rn":  container_name,   # resourceName
            "lbl": [digit],         # label
            "mbs": 10000000         # 최대 byte 크기
        }
    }

    response = requests.post(url, headers=HEADERS_CNT, json=body)
    if response.status_code in [200, 201]:
        print(f"[OK] Container '{container_name}' created at {base_url}.")
    else:
        print(f"[FAIL] Create container '{container_name}' at {base_url}: {response.text}")

# --------------------------------------------------------------------------------
# (3) Mobius 데이터 업로드 함수 (콘텐츠 인스턴스 생성)
# --------------------------------------------------------------------------------
def upload_data(container_name: str, data: str, base_url: str) -> None:
    """
    base_url + container_name 경로에 data(JSON 문자열)를 CIN으로 업로드.
    """
    url = f"{base_url}/{container_name}"
    body = {
        "m2m:cin": {
            "con": data
        }
    }

    response = requests.post(url, headers=HEADERS_CIN, json=body)
    if response.status_code in [200, 201]:
        print(f"[OK] Data uploaded to '{container_name}' at {base_url}.")
    else:
        print(f"[FAIL] Upload data to '{container_name}' at {base_url}: {response.text}")

# --------------------------------------------------------------------------------
# (4) 이미지 데이터 -> Mobius 업로드 함수
# --------------------------------------------------------------------------------
def process_and_upload(data_path: str, base_url: str) -> None:
    """
    data_path 내의 0~9 폴더(각 레이블)를 컨테이너로 생성 후, 
    이미지(.jpg)를 열어 numpy -> list 변환 -> json.dumps 해서 업로드.

    :param data_path: 실제 이미지가 들어 있는 상위 디렉토리 (예: "./data/ReducedMNIST/Train")
    :param base_url:  Mobius AE 주소 (Train용 AE or Test용 AE)
    """
    print(f"[INFO] Creating containers (digits 0~9) at {base_url} ...")
    # 1) 0~9 범위 컨테이너 생성
    for digit in range(10):
        create_container(str(digit), base_url)

    print(f"[INFO] Start uploading images from '{data_path}' to '{base_url}' ...")

    # 2) data_path 아래의 각 라벨 디렉토리 탐색
    for label in os.listdir(data_path):
        label_path = os.path.join(data_path, label)
        if os.path.isdir(label_path):
            container_name = label  # 예: "0", "1", ...
            # 해당 디렉토리 안의 jpg 파일을 순회
            for image_file in os.listdir(label_path):
                if image_file.lower().endswith(".jpg"):
                    image_path = os.path.join(label_path, image_file)

                    # Pillow로 이미지 로드 → numpy 배열 변환
                    with Image.open(image_path) as img:
                        img_array = np.array(img, dtype=np.uint8)
                        image_list = img_array.tolist()  # JSON 직렬화 가능 구조

                    # JSON 문자열 생성
                    image_data = json.dumps({"image": image_list})

                    # Mobius 업로드
                    upload_data(container_name, image_data, base_url)

    print(f"[INFO] Completed uploading images from '{data_path}' to '{base_url}'")

# --------------------------------------------------------------------------------
# (5) 메인 실행부
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    # 예시: ReducedMNIST 구조
    TRAIN_PATH = "./data/ReducedMNIST/Train"
    TEST_PATH  = "./data/ReducedMNIST/Test"

    process_and_upload(TRAIN_PATH, MOBIUS_BASE_AE_URL)

    process_and_upload(TEST_PATH, MOBIUS_TEST_AE_URL)

