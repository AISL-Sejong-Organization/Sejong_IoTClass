import csv
import requests
import os
from datetime import datetime

# --------------------------------------------------------------------------------
# Mobius 플랫폼 설정
# --------------------------------------------------------------------------------
MOBIUS_BASE_URL   = "http://203.250.148.120:20519/Mobius/"
MOBIUS_AE_NAME    = "ts-train"
MOBIUS_TEST_NAME  = "ts-test"

MOBIUS_BASE_AE_URL = os.path.join(MOBIUS_BASE_URL, MOBIUS_AE_NAME)
MOBIUS_TEST_AE_URL = os.path.join(MOBIUS_BASE_URL, MOBIUS_TEST_NAME)

# Mobius 리소스 생성 시 필요한 헤더
HEADERS_CNT = {
    'Accept':            'application/json',
    'X-M2M-RI':          '12345',
    'X-M2M-Origin':      'SOrigin',
    'Content-Type':      'application/vnd.onem2m-res+json; ty=3'
}

# Mobius 콘텐츠 인스턴스 생성 시 필요한 헤더
HEADERS_CIN = {
    'Accept':            'application/json',
    'X-M2M-RI':          '12345',
    'X-M2M-Origin':      'SOrigin',
    'Content-Type':      'application/vnd.onem2m-res+json; ty=4'
}

# --------------------------------------------------------------------------------
# 센서 컨테이너 이름
# --------------------------------------------------------------------------------
CONTAINERS = {
    "precipitation": "precipitation_sensor",
    "temperature":   "temperature_sensor",
    "wind":          "wind_sensor",
    "ground_truth":  "ground_truth"
}

# --------------------------------------------------------------------------------
# (1) Mobius 컨테이너 생성 함수
# --------------------------------------------------------------------------------
def create_container(sensor_name, base_url):
    """
    base_url 경로 아래에 sensor_name을 이름으로 하는 컨테이너를 생성한다.
    """
    url  = f"{base_url}"
    body = {
        "m2m:cnt": {
            "rn":  sensor_name,
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
# (2) Mobius 데이터 업로드 함수
# --------------------------------------------------------------------------------
def upload_data(sensor_name, data, base_url):
    """
    base_url 아래 sensor_name 컨테이너에 'data'를 콘텐츠 인스턴스(CIN) 형태로 업로드한다.
    """
    url  = f"{base_url}/{sensor_name}"
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
# (3) CSV 파일을 읽어 TRAIN/TEST 데이터 분할 후 Mobius에 업로드
# --------------------------------------------------------------------------------
def process_and_upload(csv_path):
    """
    CSV 파일(csv_path)을 읽어:
      - 70%는 TRAIN(AIOTclass-TS-Trainset) 컨테이너에 업로드
      - 나머지 30%는 TEST(AIOTclass-TS-Testset) 컨테이너에 업로드
    데이터는 CONTAINERS에 정의된 각 컨테이너에 저장한다.
    """

    # 1) 컨테이너 생성(Trainset, Testset 각각)
    for container_name in CONTAINERS.values():
        create_container(container_name, MOBIUS_BASE_AE_URL)
        create_container(container_name, MOBIUS_TEST_AE_URL)

    # 2) CSV 전체 데이터 읽기(날짜 개수)
    with open(csv_path, mode='r', encoding='utf-8') as file:
        all_rows    = list(csv.DictReader(file))
        total_days  = len(all_rows)
        cutoff_index = int(total_days * 0.7)  # 70% 지점

    # 3) 다시 한 줄씩 읽어, 70% / 30%로 분할하여 업로드
    with open(csv_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        for i, row in enumerate(reader):
            date_str = row["date"]

            # (a) precipitation 데이터 생성
            precipitation_data = {
                "date":           date_str,
                "precipitation":  float(row["precipitation"])
            }

            # (b) temperature 데이터 생성
            temperature_data = {
                "date":     date_str,
                "temp_max": float(row["temp_max"]),
                "temp_min": float(row["temp_min"])
            }

            # (c) wind 데이터 생성
            wind_data = {
                "date": date_str,
                "wind": float(row["wind"])
            }

            # (d) ground_truth 데이터 생성
            ground_truth_data = {
                "date":    date_str,
                "weather": row["weather"]
            }

            # (e) 업로드할 URL 결정 (TRAIN vs TEST)
            if i < cutoff_index:
                # TRAINSET
                upload_data(CONTAINERS["precipitation"], precipitation_data, MOBIUS_BASE_AE_URL)
                upload_data(CONTAINERS["temperature"],   temperature_data,   MOBIUS_BASE_AE_URL)
                upload_data(CONTAINERS["wind"],          wind_data,          MOBIUS_BASE_AE_URL)
                upload_data(CONTAINERS["ground_truth"],  ground_truth_data,  MOBIUS_BASE_AE_URL)
            else:
                # TESTSET
                upload_data(CONTAINERS["precipitation"], precipitation_data, MOBIUS_TEST_AE_URL)
                upload_data(CONTAINERS["temperature"],   temperature_data,   MOBIUS_TEST_AE_URL)
                upload_data(CONTAINERS["wind"],          wind_data,          MOBIUS_TEST_AE_URL)
                upload_data(CONTAINERS["ground_truth"],  ground_truth_data,  MOBIUS_TEST_AE_URL)

# --------------------------------------------------------------------------------
# (4) 메인 실행부
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    # CSV 파일 경로
    CSV_FILE_PATH = "./data/seattle-weather.csv"

    # 실제 동작
    process_and_upload(CSV_FILE_PATH)
