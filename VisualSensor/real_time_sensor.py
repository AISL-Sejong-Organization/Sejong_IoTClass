import csv
import requests
import os
import time
import json
from PIL import Image
from datetime import datetime
import numpy as np
import glob
import random

# --------------------------------------------------------------------------------
# Mobius 플랫폼 설정 상수
# --------------------------------------------------------------------------------
MOBIUS_BASE_URL    = "http://203.250.148.120:20519/Mobius/"
MOBIUS_AE_NAME     = "AIoTclass-VS-Inferencing"
MOBIUS_BASE_AE_URL = os.path.join(MOBIUS_BASE_URL, MOBIUS_AE_NAME)

# HTTP 요청 시 필요한 헤더
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
HEADERS_SUB = {
    'Accept': 'application/json',
    'X-M2M-RI': '12345',
    'X-M2M-Origin': 'SOrigin',
    'Content-Type': 'application/vnd.onem2m-res+json; ty=23'
}

# 업로드 주기(임의로 5초로 설정)
UPLOAD_INTERVAL = 5

# --------------------------------------------------------------------------------
# 센서 컨테이너 이름 매핑
# --------------------------------------------------------------------------------
CONTAINERS = {
    "image": "camera_sensor",
    # "report": "report"  # 필요 시 주석 해제
}

# --------------------------------------------------------------------------------
# (1) 컨테이너 생성 함수
# --------------------------------------------------------------------------------
def create_container(sensor_name, base_url):
    """
    Mobius에 새로운 컨테이너를 생성한다.
    :param sensor_name: 생성할 컨테이너 이름
    :param base_url: Mobius AE의 기본 URL
    """
    url = f"{base_url}"
    body = {
        "m2m:cnt": {
            "rn": sensor_name,
            "lbl": [],
            "mbs": 5000000
        }
    }

    response = requests.post(url, headers=HEADERS_CNT, json=body)
    if response.status_code in [200, 201]:
        print(f"[OK] Container '{sensor_name}' created at {base_url}.")
    else:
        print(f"[FAIL] Create container '{sensor_name}' at {base_url}: {response.text}")

# --------------------------------------------------------------------------------
# (2) 구독(subscription) 생성 함수
# --------------------------------------------------------------------------------
def create_subscription(sensor_name, base_url):
    """
    특정 컨테이너에 대한 Subscription을 생성한다.
    MQTT 주소(mqtt://203.250.148.120/...)로 데이터가 전송되도록 설정.
    :param sensor_name: 구독을 생성할 컨테이너 이름
    :param base_url: Mobius AE의 기본 URL
    """
    url = f"{base_url}/{sensor_name}"

    # 예시로 "model"이라는 subscription 리소스명을 사용
    body = {
        "m2m:sub": {
            "rn":  "model",
            "enc": {
                "net": [3]  # net=3: Create Notification
            },
            "nu":  [f"mqtt://203.250.148.120/{MOBIUS_AE_NAME}{sensor_name}?ct=json"],
            "exc": 10
        }
    }

    response = requests.post(url, headers=HEADERS_SUB, json=body)
    if response.status_code in [200, 201]:
        print(f"[OK] Subscription 'model' created for '{sensor_name}'.")
    else:
        print(f"[FAIL] Create subscription for '{sensor_name}': {response.text}")

# --------------------------------------------------------------------------------
# (3) 센서 데이터 업로드 함수
# --------------------------------------------------------------------------------
def upload_data(sensor_name, data, base_url):
    """
    해당 컨테이너(sensor_name)에 'data'를 콘텐츠 인스턴스(CIN) 형태로 업로드한다.
    :param sensor_name: 업로드할 컨테이너 이름
    :param data: 업로드할 실제 데이터(파이썬 dict)
    :param base_url: Mobius AE의 기본 URL
    """
    url = f"{base_url}/{sensor_name}"
    body = {
        "m2m:cin": {
            "con": data
        }
    }

    response = requests.post(url, headers=HEADERS_CIN, json=body)
    if response.status_code in [200, 201]:
        print(f"[OK] Data uploaded to '{sensor_name}' at {base_url}.")
    else:
        print(f"[FAIL] Upload data to '{sensor_name}' at {base_url}: {response.text}")

# --------------------------------------------------------------------------------
# (4) 이미지 디렉터리를 읽고 컨테이너 생성+구독 후 데이터를 업로드하는 메인 로직
# --------------------------------------------------------------------------------
def process_and_upload(data_path) -> None:
    """
    1) containers 목록(딕셔너리의 values())로 컨테이너 & 구독 생성
    2) data_path 아래의 모든 .jpg 파일을 찾아,
       무작위(하지만 사실상 1개뿐이면 고정) 컨테이너에 업로드
    """

    # 1) 컨테이너 & 구독 생성
    for container_name in CONTAINERS.values():
        create_container(container_name, MOBIUS_BASE_AE_URL)
        create_subscription(container_name, MOBIUS_BASE_AE_URL)

    input("\n[알림] 컨테이너와 구독 생성 완료. 계속 진행하려면 Enter...\n")

    # 2) data_path 하위 모든 폴더를 순회, .jpg만 수집
    # MNIST 내 클래스와 이미지를 무작위로 선택하는 것처럼, 여기서는 모든 이미지를 무작위 컨테이너에 업로드
    # 모든 jpg 파일 경로 가져오기
    all_images = glob.glob(os.path.join(data_path, "**/*.jpg"), recursive=True)

    # 리스트를 무작위로 섞기
    random.shuffle(all_images)

    print(f"[INFO] Found {len(all_images)} images in '{data_path}'")

    # 3) 이미지별로 무작위 컨테이너(하지만 사실 1개뿐) 선택 → 업로드
    for img_path in all_images:
        print(f"[INFO] Uploading '{img_path}' to containers...")
        with Image.open(img_path) as img:
            img_array = np.array(img, dtype=np.uint8)
            image_list = img_array.tolist()

        image_data = json.dumps({"image": image_list})

        upload_data(CONTAINERS["image"], image_data, MOBIUS_BASE_AE_URL)
        # (e) 업로드 후 대기
        time.sleep(UPLOAD_INTERVAL)

    print("[INFO] All images uploaded randomly to containers (though there's only 1).")

# --------------------------------------------------------------------------------
# 실행부
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    # CSV 파일 경로
    IMAGE_FILE_PATH = "./data/ReducedMNIST/Test"

    # 실제 동작
    process_and_upload(IMAGE_FILE_PATH)
