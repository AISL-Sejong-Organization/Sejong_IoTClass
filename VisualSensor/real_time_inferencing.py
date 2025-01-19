import os
import json
import requests
import torch
import torch.nn as nn
import numpy as np
import base64
from urllib.parse import urlparse
from torchvision import transforms
import paho.mqtt.client as mqtt
from PIL import Image
from io import BytesIO
from datetime import datetime
import time

# --------------------------------------------------------------------------------
# (1) 전역 설정값 (예: 모델 출력 크기 등)
# --------------------------------------------------------------------------------
output_size = 10

# --------------------------------------------------------------------------------
# (2) 전역 변수 (센서 데이터 저장)
# --------------------------------------------------------------------------------
sensor_data = {
    "image": None
}

# --------------------------------------------------------------------------------
# (3) MNISTCNN 모델 정의 (학습 시 사용한 모델과 동일)
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
        # 디버깅 출력은 주석 처리
        # print(f"Forward input shape: {x.shape}")
        # print(f"Forward input max: {torch.max(x)}")
        # print(f"Forward input min: {torch.min(x)}")
        x = self.pool(self.relu(self.conv1(x)))  # [batch, 32, 14, 14]
        x = self.pool(self.relu(self.conv2(x)))  # [batch, 64, 7, 7]
        x = x.view(-1, 64 * 7 * 7)               # Flatten
        x = self.relu(self.fc1(x))               # [batch, 128]
        x = self.fc2(x)                           # [batch, output_size]
        return x

# --------------------------------------------------------------------------------
# (4) Base64 모델 디코딩 함수 (필요 시)
# --------------------------------------------------------------------------------
def decode_model(base64_encoded_str, save_path="./mnist_cnn.pth"):
    """
    Base64 인코딩된 모델 문자열을 디코딩하여 로컬에 저장.
    """
    decoded_bytes = base64.b64decode(base64_encoded_str)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(decoded_bytes)
    return save_path

# --------------------------------------------------------------------------------
# (5) Mobius / OneM2M 연동을 위한 HTTP GET
# --------------------------------------------------------------------------------
def http_get(url, params=None, headers=None, iotPlatform=False):
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
    except requests.ConnectTimeout:
        print(f"[Mobius Error] Connection timed out for URL: {url}")
        return None
    except requests.HTTPError as http_err:
        print(f"[Mobius Error] HTTP error occurred for URL {url}: {http_err}")
        return None
    except Exception as err:
        print(f"[Mobius Error] An error occurred for URL {url}: {err}")
        return None

# --------------------------------------------------------------------------------
# (6) Mobius 연동: 특정 컨테이너 URL 하위의 모든 CIN(con) 가져오기
# --------------------------------------------------------------------------------
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

# --------------------------------------------------------------------------------
# (7) Mobius에서 (0~9) 컨테이너에 있는 이미지 데이터를 가져오기
# --------------------------------------------------------------------------------
def fetch_data(base_url, container_name):
    """
    - base_url 아래 특정 컨테이너 접근 -> 모든 CIN con 수집
    - 리턴: {'images': [...], 'labels': [...]}
    """
    data = {'images': [], 'labels': []}
    url = f"{base_url}/{container_name}"
    con_list = all_cin_get_uri(url)
    
    for con in con_list:
        # con = {"image": "..."} 형태라고 가정
        if "image" in con:
            data['images'].append(con['image'])  # Base64 문자열 또는 리스트 형태
            data['labels'].append(int(container_name))  # label은 컨테이너 이름을 정수로 변환
        else:
            print(f"[WARN] 'image' key not found in content: {con}")
    return data


# --------------------------------------------------------------------------------
# (8) 이미지 전처리 함수 (Base64 → 텐서)
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
# (9) MQTT 관련 콜백
# --------------------------------------------------------------------------------
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected OK")
    else:
        print("Bad connection Returned code=", rc)

def on_disconnect(client, userdata, rc=0):
    print(f"Disconnected with return code {rc}")

def on_subscribe(client, userdata, mid, granted_qos):
    print(f"Subscribed: mid={mid}, granted_qos={granted_qos}")

def create_on_message_callback(sensor_name):
    """
    sensor_name(예: 'image')를 기억해두고,
    실제 메시지를 받았을 때 해당 센서 이름으로 sensor_data에 저장
    """
    def on_message(client, userdata, msg):
        payload_str = msg.payload.decode("utf-8")
        try:
            payload_dict = json.loads(payload_str)
        except Exception as ex:
            print("Payload parsing error:", ex)
            return

        try:
            # 메시지 구조에 따라 경로 조정
            # 예시 구조: {'pc': {'m2m:sgn': {'nev': {'rep': {'m2m:cin': {'con': {'image': '...'}}}}}}}}
            content = payload_dict['pc']['m2m:sgn']['nev']['rep']['m2m:cin']['con']
            print(f"[{sensor_name}] Received content: {content}")

            # 전역 sensor_data에 저장
            sensor_data[sensor_name] = content[sensor_name]

        except KeyError:
            print("KeyError: 메시지 구조가 예상과 다릅니다.")
            pass

    return on_message

def subscribing(ip, port, topic, sensor_name):
    """
    특정 IP/port/topic으로 MQTT 구독. loop_start()로 백그라운드 메시지 처리
    """
    client = mqtt.Client()
    client.on_connect    = on_connect
    client.on_disconnect = on_disconnect
    client.on_subscribe  = on_subscribe
    client.on_message    = create_on_message_callback(sensor_name)

    try:
        client.connect(ip, port)
    except Exception as e:
        print(f"[MQTT Error] Unable to connect to MQTT broker at {ip}:{port}: {e}")
        return None

    client.subscribe('/oneM2M/req/+/' + topic + '/#', qos=1)
    client.loop_start()

    return client

# --------------------------------------------------------------------------------
# (10) Mobius에 등록된 subscription 정보 가져오기
# --------------------------------------------------------------------------------
def get_iot_platform_sub_nu(url):
    sub_dict = http_get(url, iotPlatform=True)
    if sub_dict and 'm2m:sub' in sub_dict and 'nu' in sub_dict['m2m:sub']:
        return sub_dict['m2m:sub']['nu']
    else:
        print(f"[Mobius Error] 'nu' not found in subscription data from {url}")
        return []

def sub_iot_platform_cin_con(nu, port, sensor_name):
    # nu 예: "mqtt://203.250.xxx.xxx/camera_sensor"
    parsed = urlparse(nu)
    ip = parsed.hostname
    topic = parsed.path.strip('/')  # "camera_sensor"

    if not topic:
        print(f"[Mobius Error] Topic not found in 'nu': {nu}")
        return None

    client = subscribing(ip, port, topic, sensor_name)
    return client

# --------------------------------------------------------------------------------
# (11) Mobius 연결 설정
# --------------------------------------------------------------------------------
CONTAINERS = {
    "image": "camera_sensor",
    # "report": "report"  # 필요 시 주석 해제
}
MOBIUS_BASE_URL    = "http://203.250.148.120:20519/Mobius/"
MOBIUS_AE_NAME     = "AIoTclass-VS-Inferencing"
MOBIUS_BASE_AE_URL = os.path.join(MOBIUS_BASE_URL, MOBIUS_AE_NAME)
MOBIUS_MODEL_REPOSITORY_CNT_NAME  = "model_repository"       # 모델 저장소 컨테이너 이름
MOBIUS_SENSOR_DATA_CNT_NAME       = "sensor_data"           # 센서 데이터 컨테이너 이름
MOBIUS_INFERENCING_RESULT_CNT_NAME= "inferencing_result"    # 추론 결과 컨테이너 이름
MQTT_PORT          = 20516
HEADERS_CIN = {
    'Accept': 'application/json',
    'X-M2M-RI': '12345',
    'X-M2M-Origin': 'SOrigin',
    'Content-Type': 'application/vnd.onem2m-res+json; ty=4'
}

# --------------------------------------------------------------------------------
# (12) 메인 실행부
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    # 1) 모델 로드: 로컬에서 모델 파일을 직접 불러옵니다.
    model_path = "./mnist_cnn.pth"
    if not os.path.exists(model_path):
        print(f"[Error] Model file '{model_path}' does not exist.")
        exit(1)

    model = MNISTCNN(output_size=output_size)
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        print(f"[MODEL SERVER] Loaded model from '{model_path}'")
    except Exception as e:
        print(f"[Error] Failed to load model: {e}")
        exit(1)

    # 2) MQTT 클라이언트 설정 및 구독
    clients = []
    for sensor_key, container in CONTAINERS.items():
        # 예: http://.../Mobius/testman11/camera_sensor/model
        sub_instance_url = os.path.join(MOBIUS_BASE_AE_URL, container, "model")
        sub_nu_list = get_iot_platform_sub_nu(sub_instance_url)  # ["mqtt://ip/camera_sensor", ...]

        for nu in sub_nu_list:
            client = sub_iot_platform_cin_con(nu, MQTT_PORT, sensor_key)
            if client:
                clients.append(client)

    if clients:
        print("모든 센서 구독이 설정되었습니다. 백그라운드에서 메시지를 받습니다.\n")
    else:
        print("[Warning] No MQTT clients were subscribed.")

    # 3) 이미지 전처리 및 추론 루프
    # 학습 시 사용했던 전처리를 동일하게 적용
    # transforms.ToTensor()를 사용하지 않도록 수정
    transform = transforms.Resize((28, 28))  # MNIST 크기로 리사이즈

    try:
        while True:
            if sensor_data["image"]:
                # 4) 이미지 데이터가 있으면 모델 추론
                image = sensor_data["image"]
                sensor_data["image"] = None

                # 가정: image 변수가 Base64 문자열 또는 리스트 형태
                if isinstance(image, str):
                    # Base64 문자열인 경우
                    try:
                        image_tensor = preprocess_image_bytes(image)  # (1, 1, 28, 28)
                        if image_tensor is None:
                            continue
                        # print(f"Image tensor shape: {image_tensor.shape}")  # Should be [1, 1, 28, 28]
                        # print(f"Max value: {torch.max(image_tensor)}")      # Should be <= 255.0
                        # print(f"Min value: {torch.min(image_tensor)}")      # Should be >= 0.0
                    except Exception as e:
                        print(f"[Error] Image preprocessing failed: {e}")
                        continue
                elif isinstance(image, list):
                    # 리스트 형태인 경우 (예: 픽셀 값 리스트)
                    try:
                        image_np = np.array(image, dtype=np.uint8)    # (H, W) 또는 (H, W, C)
                        image_pil = Image.fromarray(image_np).convert("L")  # 흑백
                        # transforms 적용
                        image_pil = transform(image_pil)
                        arr = np.array(image_pil).astype(np.float32)
                        image_tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, 28, 28)
                        # print(f"Image tensor shape: {image_tensor.shape}")  # Should be [1, 1, 28, 28]
                        # print(f"Max value: {torch.max(image_tensor)}")      # Should be <= 255.0
                        # print(f"Min value: {torch.min(image_tensor)}")      # Should be >= 0.0
                    except Exception as e:
                        print(f"[Error] Image conversion failed: {e}")
                        continue
                else:
                    print("[Error] Unsupported image format.")
                    continue

                # 5) 추론 결과를 출력 및 Mobius에 업로드
                with torch.no_grad():
                    output = model(image_tensor)
                    _, predicted = torch.max(output.data, 1)
                    pred = predicted.item()
                    print(f"[MODEL SERVER] Prediction: {pred}")

                # (4.2) 추론 결과를 Mobius의 inferencing_result 컨테이너에 업로드
                result_data = {
                    "prediction": pred,
                    "timestamp": datetime.now().isoformat()
                }

                # (4.3) 응답(추론 결과) 소켓으로 전송
                # 기존 코드에서는 소켓을 통해 결과를 전송하는 부분이 있었으나, 
                # 현재 스크립트에서는 소켓 서버가 없으므로 이 부분은 생략하거나 
                # 필요 시 추가 구현해야 합니다.

            time.sleep(1)  # 짧은 대기 시간으로 루프 최적화

    except KeyboardInterrupt:
        print("\n프로그램 종료 요청됨. MQTT loop 중단.")
        for c in clients:
            c.loop_stop()
        print("종료합니다.")
