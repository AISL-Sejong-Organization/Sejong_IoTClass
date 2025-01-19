import csv
import requests
import os
import time
from datetime import datetime

# --------------------------------------------------------------------------------
# Mobius 플랫폼 설정 상수
# --------------------------------------------------------------------------------
MOBIUS_BASE_URL    = "http://203.250.148.120:20519/Mobius/"
MOBIUS_AE_NAME     = "AIoTclass-TS-Inferencing"
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

# 업로드 주기(임의로 3초로 설정)
UPLOAD_INTERVAL = 5

# CSV 데이터 window_size (현재 예시 코드 내에서 직접 활용하는 로직은 없음)
window_size = 5

# --------------------------------------------------------------------------------
# 센서 컨테이너 이름 매핑
# --------------------------------------------------------------------------------
CONTAINERS = {
    "precipitation": "precipitation_sensor",
    "temperature":   "temperature_sensor",
    "wind":          "wind_sensor",
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
# (4) CSV를 읽고 컨테이너 생성+구독 후 데이터를 업로드하는 메인 로직
# --------------------------------------------------------------------------------
def process_and_upload(csv_path):
    """
    CSV 파일을 읽어서 각 날짜에 대한 센서 데이터(precipitation, temperature, wind 등)를 
    Mobius 컨테이너에 업로드하는 함수. 컨테이너 및 구독도 자동으로 생성한다.
    :param csv_path: 업로드할 CSV 파일 경로
    """

    # 1) 컨테이너 & 구독 생성
    for container_name in CONTAINERS.values():
        create_container(container_name, MOBIUS_BASE_AE_URL)
        create_subscription(container_name, MOBIUS_BASE_AE_URL)

    input("\n[알림] 컨테이너와 구독이 생성되었습니다. 계속하려면 Enter를 누르세요...\n")

    # 2) CSV 파일 전체 행수를 확인 (날짜 범위 계산용)
    with open(csv_path, mode='r', encoding='utf-8') as file:
        reader = list(csv.DictReader(file))
        total_days = len(reader)  # 전체 데이터 수
    print(f"[INFO] 총 {total_days}일치 데이터가 '{csv_path}'에 있습니다.")

    # 3) CSV 파일에서 한 줄씩 읽어 데이터 업로드
    with open(csv_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        for i, row in enumerate(reader):
            date_str = row["date"]  # 예: '2012-10-01'

            # (a) Precipitation 데이터
            precipitation_data = {
                "date":           date_str,
                "precipitation":  float(row["precipitation"])
            }

            # (b) Temperature 데이터
            temperature_data = {
                "date":     date_str,
                "temp_max": float(row["temp_max"]),
                "temp_min": float(row["temp_min"])
            }

            # (c) Wind 데이터
            wind_data = {
                "date": date_str,
                "wind": float(row["wind"])
            }

            # (d) 실제 업로드 (각 센서 컨테이너에 PUT)
            upload_data(CONTAINERS["precipitation"], precipitation_data, MOBIUS_BASE_AE_URL)
            upload_data(CONTAINERS["temperature"],   temperature_data,   MOBIUS_BASE_AE_URL)
            upload_data(CONTAINERS["wind"],          wind_data,          MOBIUS_BASE_AE_URL)

            # (e) 업로드 후 대기
            time.sleep(UPLOAD_INTERVAL)

            # (선택) 중간 진행도 출력
            # print(f"[INFO] Data uploaded for {date_str} ({i+1}/{total_days} done)")

            # (추가 예시) 보고(report) 컨테이너 업로드
            # if (i + 1) % 5 == 0:
            #     report = {"date": date_str, "weather": row["weather"]}
            #     upload_data(CONTAINERS["report"], report, MOBIUS_BASE_AE_URL)

# --------------------------------------------------------------------------------
# 실행부
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    # CSV 파일 경로
    CSV_FILE_PATH = "./data/seattle-weather.csv"

    # 실제 동작
    process_and_upload(CSV_FILE_PATH)
