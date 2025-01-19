import os
import socket
import torch
import torch.nn as nn
import numpy as np
import base64
import json
import requests
from PIL import Image
from io import BytesIO
from datetime import datetime
from urllib.parse import urlparse

# --------------------------------------------------------------------------------
# (1) Mobius 관련 설정
# --------------------------------------------------------------------------------
MOBIUS_BASE_URL = "http://203.250.148.120:20519/Mobius/"
MOBIUS_AE_NAME  = "test-OC"

MOBIUS_SENSOR_DATA_CNT_NAME       = "sensor_data"
MOBIUS_INFERENCING_RESULT_CNT_NAME= "inferencing_result"
MOBIUS_MODEL_REPOSITORY_CNT_NAME  = "model_repository"

MOBIUS_BASE_AE_URL = os.path.join(MOBIUS_BASE_URL, MOBIUS_AE_NAME)

# Mobius 업로드 시 사용될 헤더 (콘텐츠 인스턴스 생성)
HEADERS_CIN = {
    'Accept': 'application/json',
    'X-M2M-RI': '12345',
    'X-M2M-Origin': 'SOrigin',
    'Content-Type': 'application/vnd.onem2m-res+json; ty=4'
}

# --------------------------------------------------------------------------------
# (2) Mobius 연동 함수
# --------------------------------------------------------------------------------
def http_get(url, params=None, headers=None, iotPlatform=False):
    """
    - iotPlatform=True이면 OneM2M 관련 헤더 자동 적용
    - GET 요청 후 JSON 반환
    """
    if iotPlatform:
        headers = {
            'Accept': 'application/json',
            'X-M2M-RI': '12345',
            'X-M2M-Origin': 'SOrigin'
        }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[Mobius Error] GET {url} : {e}")
        return None


def all_cin_get_uri(path, max_retries=5):
    """
    - path + '?fu=1&ty=4'로 모든 CIN URI 가져옴
    - 각각에 대해 GET → 'con' 필드 수집
    """
    path += "?fu=1&ty=4"
    parsed = urlparse(path)
    base_path = f"{parsed.scheme}://{parsed.netloc}/"

    con_list = []
    all_uri  = http_get(path, iotPlatform=True)
    if not all_uri or "m2m:uril" not in all_uri:
        return con_list

    for uri in all_uri["m2m:uril"]:
        retries = 0
        while retries < max_retries:
            cin_json = http_get(base_path + uri, iotPlatform=True)
            if cin_json is not None and "m2m:cin" in cin_json:
                con = cin_json["m2m:cin"]["con"]
                con_list.append(con)
                break
            else:
                retries += 1
                print(f"[Mobius] Retry {retries} for {base_path + uri}")
    return con_list


def decode_model(base64_str, save_path="./mnist_cnn.pth"):
    """
    - Base64 문자열을 받아 .pth 모델 파일로 저장
    - 저장 경로 반환
    """
    decoded_bytes = base64.b64decode(base64_str)
    with open(save_path, "wb") as f:
        f.write(decoded_bytes)
    return save_path


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
# (3) 모델 정의 (예: MNISTCNN)
# --------------------------------------------------------------------------------
class MNISTCNN(nn.Module):
    def __init__(self, output_size=10):
        super(MNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1   = nn.Linear(64 * 7 * 7, 128)
        self.fc2   = nn.Linear(128, output_size)
        self.pool  = nn.MaxPool2d(2, 2)
        self.relu  = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --------------------------------------------------------------------------------
# (4) 이미지 전처리 함수 (Base64 → 텐서)
# --------------------------------------------------------------------------------
def preprocess_image_bytes(b64_str):
    """
    - Base64 문자열(b64_str)을 받아,
      1) base64 디코딩 → 2) BytesIO → 3) PIL.Image
      4) 28×28 흑백으로 변환 → 5) NumPy → 6) PyTorch 텐서
    - 최종 shape: (1, 1, 28, 28)
    """
    try:
        # 1) base64 → bytes
        img_bytes = base64.b64decode(b64_str)
        # 2) BytesIO → PIL
        pil_image = Image.open(BytesIO(img_bytes)).convert("L")
        # 3) 28×28 리사이즈 (MNIST)
        pil_image = pil_image.resize((28, 28))
        # 4) [0~255] 범위 유지 (정규화 제거)
        arr = np.array(pil_image).astype(np.float32)
        # 5) (H, W) → (1, 1, H, W)
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1, 1, 28, 28]
        return tensor
    except Exception as e:
        print(f"[Error] preprocess_image_bytes failed: {e}")
        return None

# --------------------------------------------------------------------------------
# (5) 메인: 소켓 서버 + Mobius 모델 로드
# --------------------------------------------------------------------------------
if __name__ == "__main__":

    # 1) Mobius에서 최신 모델 가져오기
    model_repo_path = os.path.join(MOBIUS_BASE_AE_URL, MOBIUS_MODEL_REPOSITORY_CNT_NAME)
    con_list = all_cin_get_uri(model_repo_path)
    if not con_list:
        print("[Error] No model data in Mobius.")
        exit(1)

    # 1.1) 가장 최근에 올라온 모델(con_list[-1]) 사용
    latest_model_data = con_list[-1]  # {"metadata": {...}, "model_file": "...(base64)..." }
    base64_model_str  = latest_model_data["model_file"]
    metadata          = latest_model_data["metadata"]

    # 1.2) 디코딩 → 로컬에 저장
    model_path = decode_model(base64_model_str, save_path="./mnist_cnn.pth")

    # 1.3) 모델 로드
    model = MNISTCNN(output_size=10)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    print(f"[MODEL SERVER] Loaded model: {metadata['model_name']} (version {metadata['version']})")

    # 2) 소켓 서버 준비
    host = '127.0.0.1'
    port = 5000

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)
    print(f"[MODEL SERVER] Listening on {host}:{port}")

    # 3) 연결 대기 → 연결 시 데이터 수신/추론
    while True:
        conn, addr = server_socket.accept()
        print(f"[MODEL SERVER] Connected by {addr}")

        while True:
            # 4) 센서(또는 클라이언트)로부터 메시지 수신 (JSON string)
            data = conn.recv(4096).decode('utf-8')
            if not data:
                break

            try:
                msg = json.loads(data)
            except Exception as e:
                print("[MODEL SERVER] JSON decode error:", e)
                break

            # (4.1) 센서 데이터를 Mobius의 sensor_data 컨테이너에 업로드
            sensor_data_con = {
                "timestamp": datetime.now().isoformat(),
                "raw_base64": msg.get("image_data", ""),  # 원한다면 Base64를 그대로 보관
            }
            upload_data(MOBIUS_SENSOR_DATA_CNT_NAME, sensor_data_con, MOBIUS_BASE_AE_URL)

            # (4.2) 추론
            b64_str    = msg.get("image_data", "")
            image_tensor = preprocess_image_bytes(b64_str)

            with torch.no_grad():
                output = model(image_tensor)
                pred   = torch.argmax(output).item()

            # (4.3) 추론 결과
            result_data = {
                "prediction": pred,
                "timestamp": datetime.now().isoformat()
            }

            # (4.4) 추론 결과를 Mobius의 inferencing_result 컨테이너에 업로드
            upload_data(MOBIUS_INFERENCING_RESULT_CNT_NAME, result_data, MOBIUS_BASE_AE_URL)

            # (4.5) 응답(추론 결과) 소켓으로 전송
            print(f"[MODEL SERVER] Prediction: {pred}")
            result_json = json.dumps(result_data)
            conn.sendall(result_json.encode('utf-8'))

        # 연결 종료
        conn.close()
        print(f"[MODEL SERVER] Connection closed by {addr}")
