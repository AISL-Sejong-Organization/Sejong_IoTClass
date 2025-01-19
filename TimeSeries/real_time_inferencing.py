import os
import json
import requests
import torch
import torch.nn as nn
import numpy as np
from urllib.parse import urlparse
import paho.mqtt.client as mqtt
from sklearn.preprocessing import MinMaxScaler
import time

# --------------------------------------------------------------------------------
# (1) 센서 데이터 -> Numpy 변환 함수
# --------------------------------------------------------------------------------
def transform_sensor_data(sensor_data):
    """
    sensor_data 예시:
    {
      'precipitation': [
        {'date': '2012-01-01', 'precipitation': 0},
        {'date': '2012-01-02', 'precipitation': 10.9},
        ...
      ],
      'temperature': [
        {'date': '2012-01-01', 'temp_max': 12.8, 'temp_min': 5},
        ...
      ],
      'wind': [
        {'date': '2012-01-01', 'wind': 4.7},
        ...
      ]
    }

    결과 (dict):
    {
      'precipitation': array([[0], [10.9], [0.8], ...]),    # shape: (N,1)
      'temperature':   array([[12.8, 5], [10.6, 2.8], ...]),# shape: (N,2)
      'wind':          array([[4.7], [4.5], [2.3], ...])    # shape: (N,1)
    }
    """
    # 1) 센서별 리스트에 (sub)array로 변환
    precipitation_list = []
    temperature_list   = []
    wind_list          = []

    # precipitation -> (N,1)
    for item in sensor_data['precipitation']:
        precipitation_list.append([item['precipitation']])

    # temperature -> (N,2)
    for item in sensor_data['temperature']:
        temperature_list.append([item['temp_max'], item['temp_min']])

    # wind -> (N,1)
    for item in sensor_data['wind']:
        wind_list.append([item['wind']])

    # 2) numpy 배열로 변환
    precipitation_array = np.array(precipitation_list, dtype=np.float32)
    temperature_array   = np.array(temperature_list,   dtype=np.float32)
    wind_array          = np.array(wind_list,          dtype=np.float32)

    # 3) 결과 딕셔너리
    transformed_data = {
        'precipitation': precipitation_array,  # shape: (N,1)
        'temperature':   temperature_array,    # shape: (N,2)
        'wind':          wind_array            # shape: (N,1)
    }

    return transformed_data


# --------------------------------------------------------------------------------
# (2) 추론용 슬라이딩 윈도우 구성 함수
# --------------------------------------------------------------------------------
def make_inference_input(transformed_data, window_size):
    """
    transformed_data: {
      'precipitation': (N,1),
      'temperature':   (N,2),
      'wind':          (N,1)
    }

    - 마지막 window_size만 추출
    - (window_size, 1) + (window_size, 2) + (window_size, 1) = (window_size, 4)
    - MinMaxScaler 간단히 fit_transform (실제 프로덕션에서는 학습 시점 scaler 사용 권장)
    - 최종 (1, window_size, 4) 형태 반환 (배치 차원 추가)
    """
    precipitation = transformed_data["precipitation"]
    temperature   = transformed_data["temperature"]
    wind          = transformed_data["wind"]

    N = precipitation.shape[0]
    if N < window_size:
        return None  # 데이터가 부족하면 None 반환

    # 마지막 window_size 슬라이싱
    prec_win = precipitation[-window_size:]  # (window_size,1)
    temp_win = temperature[-window_size:]    # (window_size,2)
    wind_win = wind[-window_size:]           # (window_size,1)

    # 간단히 개별 fit_transform
    scaler_prec = MinMaxScaler()
    scaler_temp = MinMaxScaler()
    scaler_wind = MinMaxScaler()

    prec_scaled = scaler_prec.fit_transform(prec_win)
    temp_scaled = scaler_temp.fit_transform(temp_win)
    wind_scaled = scaler_wind.fit_transform(wind_win)

    # 열(특성) 단위로 결합 → (window_size, 4)
    features = []
    for i in range(window_size):
        row = np.concatenate([prec_scaled[i], temp_scaled[i], wind_scaled[i]], axis=0)
        features.append(row)

    features_array = np.array(features, dtype=np.float32)   # (window_size, 4)
    features_array = np.expand_dims(features_array, axis=0) # (1, window_size, 4)

    return features_array


# --------------------------------------------------------------------------------
# (3) LSTM 모델 정의
# --------------------------------------------------------------------------------
class WeatherLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(WeatherLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc   = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        out    = self.fc(out[:, -1, :])  
        return out  # (batch, output_size)


# --------------------------------------------------------------------------------
# (4) 전역 변수 (센서 데이터 저장)
# --------------------------------------------------------------------------------
sensor_data = {
    "precipitation": [],
    "temperature": [],
    "wind": []
}


# --------------------------------------------------------------------------------
# (5) Mobius / OneM2M 연동을 위한 HTTP GET
# --------------------------------------------------------------------------------
def http_get(url, params=None, headers=None, iotPlatform=None):
    if iotPlatform:
        headers = {
            'Accept': 'application/json',
            'X-M2M-RI': '12345',
            'X-M2M-Origin': 'SOrigin'
        }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        return json.loads(response.text)

    except requests.ConnectTimeout:
        print(f"Connection timed out for URL: {url}")
        return None

    except requests.HTTPError as http_err:
        print(f"HTTP error occurred for URL {url}: {http_err}")
        return None

    except Exception as err:
        print(f"An error occurred for URL {url}: {err}")
        return None


# --------------------------------------------------------------------------------
# (6) MQTT 관련 콜백
# --------------------------------------------------------------------------------
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("connected OK")
    else:
        print("Bad connection Returned code=", rc)

def on_disconnect(client, userdata, flags, rc=0):
    print(str(rc))

def on_subscribe(client, userdata, mid, granted_qos):
    print("subscribed: " + str(mid) + " " + str(granted_qos))


def create_on_message_callback(sensor_name):
    """
    sensor_name(예: 'precipitation')을 기억해두고,
    실제 메시지를 받았을 때 해당 센서 이름으로 sensor_data에 누적
    """
    def on_message(client, userdata, msg):
        payload_str = msg.payload.decode("utf-8")
        try:
            payload_dict = eval(payload_str)  # 또는 json.loads(payload_str)
        except Exception as ex:
            print("Payload parsing error:", ex)
            return

        try:
            content = payload_dict['pc']['m2m:sgn']['nev']['rep']['m2m:cin']['con']
            print(f"[{sensor_name}] Received content: {content}")

            # 전역 sensor_data에 누적
            sensor_data[sensor_name].append(content)

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

    client.connect(ip, port)
    client.subscribe('/oneM2M/req/+/' + topic + '/#', 1)
    client.loop_start()

    return client


# --------------------------------------------------------------------------------
# (7) Mobius에 등록된 subscription 정보 가져오기
# --------------------------------------------------------------------------------
def get_iot_platform_sub_nu(url):
    sub_dict = http_get(url, iotPlatform=True)
    sub_nu = sub_dict['m2m:sub']["nu"]
    return sub_nu


def sub_iot_platform_cin_con(nu, port, sensor_name):
    # nu 예: "mqtt://203.250.xxx.xxx/temperature_sensor"
    ip = urlparse(nu).hostname
    path_segments = urlparse(nu).path.split('/')
    topic = next(segment for segment in path_segments if segment)
    port = int(port)

    client = subscribing(ip, port, topic, sensor_name)
    return client


# --------------------------------------------------------------------------------
# (8) Mobius 연결 설정
# --------------------------------------------------------------------------------
CONTAINERS = {
    "precipitation": "precipitation_sensor",
    "temperature":   "temperature_sensor",
    "wind":          "wind_sensor",
}

MOBIUS_BASE_URL    = "http://203.250.148.120:20519/Mobius/"
MOBIUS_AE_NAME     = "AIoTclass-TS-Inferencing"
MOBIUS_BASE_AE_URL = os.path.join(MOBIUS_BASE_URL, MOBIUS_AE_NAME)
MQTT_PORT          = 20516


# --------------------------------------------------------------------------------
# (9) 메인 실행부
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    # 1) 여러 센서를 동시에 구독하기 위해 MQTT Client를 각각 생성
    clients = []
    for sensor_key, container in CONTAINERS.items():
        # 예: http://.../Mobius/testman5/temperature_sensor/model
        sub_instance_url = os.path.join(MOBIUS_BASE_AE_URL, container, "model")
        sub_nu_list = get_iot_platform_sub_nu(sub_instance_url)  # ["mqtt://ip/temperature_sensor", ...]

        for nu in sub_nu_list:
            client = sub_iot_platform_cin_con(nu, MQTT_PORT, sensor_key)
            clients.append(client)

    print("모든 센서 구독이 설정되었습니다. 백그라운드에서 메시지를 받습니다.\n")

    # 2) 모델 준비
    input_size  = 4   # precipitation=1, temperature=2, wind=1 → 총 4
    hidden_size = 5
    output_size = 5   # 예: 5가지 날씨 클래스라고 가정
    window_size = 5   # 최근 5개 → 다음 1개 예측

    model = WeatherLSTM(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load("./weather_lstm_model.pth"))
    model.eval()

    try:
        while True:
            # 3) 데이터가 window_size개 이상 쌓였는지 체크
            n_prec = len(sensor_data["precipitation"])
            n_temp = len(sensor_data["temperature"])
            n_wind = len(sensor_data["wind"])

            if (n_prec >= window_size) and (n_temp >= window_size) and (n_wind >= window_size):
                # 3.1) 전역 sensor_data -> numpy 변환
                transformed = transform_sensor_data(sensor_data)

                # 3.2) 마지막 window_size만큼 슬라이딩 윈도우 만들기
                input_array = make_inference_input(transformed, window_size)
                if input_array is not None:
                    input_tensor = torch.tensor(input_array, dtype=torch.float32)

                    # 3.3) 모델 추론
                    with torch.no_grad():
                        outputs = model(input_tensor)  # shape: (1, output_size)
                    predicted_class = torch.argmax(outputs, dim=1).item()
                    categories = ['drizzle', 'fog', 'rain', 'snow', 'sun']
                    predicted_class = categories[predicted_class]

                    # 3.4) 예측 결과 출력
                    print(f"[Inference] Last {window_size} days → Next day predicted class = {predicted_class}")
                time.sleep(3)

    except KeyboardInterrupt:
        print("\n프로그램 종료 요청됨. MQTT loop 중단.")
        for c in clients:
            c.loop_stop()
        print("종료합니다.")
