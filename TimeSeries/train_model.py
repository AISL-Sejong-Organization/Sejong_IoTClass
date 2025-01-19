import csv
import requests
import os
import numpy as np
import torch
import torch.nn as nn
import json
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from urllib.parse import urlparse

# --------------------------------------------------------------------------------
# 시드 고정 (재현성)
# --------------------------------------------------------------------------------
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# --------------------------------------------------------------------------------
# 모델 파라미터
# --------------------------------------------------------------------------------
hidden_size   = 5
window_size   = 3
batch_size    = 4
num_epochs    = 100
learning_rate = 0.001

# --------------------------------------------------------------------------------
# Mobius 플랫폼 설정
# --------------------------------------------------------------------------------
MOBIUS_BASE_URL    = "http://203.250.148.120:20519/Mobius/"
MOBIUS_AE_NAME     = "AIOTclass-TS-Trainset"
MOBIUS_TEST_NAME   = "AIOTclass-TS-Testset"

MOBIUS_BASE_AE_URL  = os.path.join(MOBIUS_BASE_URL, MOBIUS_AE_NAME)
MOBIUS_TEST_AE_URL  = os.path.join(MOBIUS_BASE_URL, MOBIUS_TEST_NAME)
HEADERS_GET = {
    'Accept':       'application/json',
    'X-M2M-RI':     '12345',
    'X-M2M-Origin': 'SOrigin'
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
# (1) HTTP 요청 및 데이터 가져오기
# --------------------------------------------------------------------------------
def http_get(url, params=None, headers=None, iotPlatform=None):
    """
    주어진 URL에 GET 요청을 보내고, 결과를 JSON(dict)로 반환한다.
    iotPlatform=True이면 OneM2M용 기본 헤더가 적용된다.
    """
    if iotPlatform:
        headers = {
            'Accept':       'application/json',
            'X-M2M-RI':     '12345',
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

def all_cin_get_uri(path, max_retries=10):
    """
    주어진 컨테이너 경로에서 모든 콘텐츠 인스턴스(CIN) URI를 얻어
    각각을 GET하여 'con' 필드를 모아 리스트로 반환한다.
    """
    path = path + '?fu=1&ty=4'  # 모든 CIN(ResourceType=4) 조회 쿼리
    parsed_path = urlparse(path)
    base_path   = f"{parsed_path.scheme}://{parsed_path.netloc}/"

    con_list = []
    all_uri  = http_get(path, iotPlatform=True)
    if not all_uri:
        return con_list  # 에러 시 빈 리스트 반환

    # 각 URI에 대해 실제 데이터 GET
    for uri in all_uri["m2m:uril"]:
        retries = 0
        while retries < max_retries:
            cin = http_get(base_path + uri, iotPlatform=True)
            if cin is not None:
                con_list.append(cin["m2m:cin"]["con"])
                break
            else:
                retries += 1
                print(f"[Retry {retries}] for URL: {base_path + uri}")

        if retries == max_retries:
            print(f"[FAIL] Data not fetched after {max_retries} attempts for URL: {base_path + uri}")

    return con_list

def fetch_data(base_url):
    """
    base_url 하위의 각 컨테이너(CONTAINERS)로부터 데이터를 가져와 dict 형태로 반환한다.
    ground_truth 컨테이너는 weather 필드만 추출한다.
    """
    data = {key: [] for key in CONTAINERS.keys()}
    for key, container in CONTAINERS.items():
        url = f"{base_url}/{container}"
        con_list = all_cin_get_uri(url)

        # 가져온 con_list를 data dict에 저장
        for con in con_list:
            if key == "ground_truth":
                # 'weather' 필드만 추출
                data[key].append(con['weather'])
            else:
                # date 이외의 필드를 리스트로 묶어서 저장
                data[key].append([con[k] for k in con.keys() if k != "date"])
    return data

# --------------------------------------------------------------------------------
# (2) 데이터 전처리(슬라이딩 윈도우, 스케일링, 원-핫 인코딩 등)
# --------------------------------------------------------------------------------
def preprocess_data(data, window_size):
    """
    data(dict)에서 precipitation, temperature, wind, ground_truth를 가져와
    - MinMaxScaler로 스케일링
    - weather(문자열)에 대해 OneHotEncoder 적용
    - window_size만큼 시계열 슬라이싱
    """
    # (a) NumPy 배열 변환
    precipitation = np.array(data["precipitation"])
    temperature   = np.array(data["temperature"])
    wind          = np.array(data["wind"])
    weather       = np.array(data["ground_truth"])  # 문자열 배열

    # (b) MinMax 정규화
    scaler = MinMaxScaler()
    precipitation = scaler.fit_transform(precipitation)
    temperature   = scaler.fit_transform(temperature)
    wind          = scaler.fit_transform(wind)

    # (c) One-hot 인코딩(날씨 레이블)
    encoder         = OneHotEncoder(sparse_output=False)
    weather_encoded = encoder.fit_transform(weather.reshape(-1, 1))
    
    # # 클래스(레이블) 목록 출력
    print("Categories:", encoder.categories_)  # (O)
    # 예: Categories: [array(['drizzle', 'fog', 'rain', 'snow', 'sun'], dtype='<U7')]

    # (d) 슬라이딩 윈도우(X, y) 생성
    X, y = [], []
    for i in range(len(precipitation) - window_size):
        features = np.concatenate([
            precipitation[i:i+window_size],
            temperature[i:i+window_size],
            wind[i:i+window_size]
        ], axis=1)
        X.append(features)
        y.append(weather_encoded[i+window_size])

    return np.array(X), np.array(y), scaler, encoder

# --------------------------------------------------------------------------------
# (3) 모델 정의 (LSTM)
# --------------------------------------------------------------------------------
class WeatherLSTM(nn.Module):
    """
    입력 시퀀스로부터 날씨 분류(One-hot 형태)를 수행하는 간단한 LSTM 모델.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(WeatherLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc   = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        x.shape: (batch, seq_len, input_size)
        LSTM 출력 중 마지막 타임스텝(hidden state)을 fully connected로 연결.
        """
        out, _ = self.lstm(x)
        out    = self.fc(out[:, -1, :])  # 마지막 타임스텝의 출력만 사용
        return out

# --------------------------------------------------------------------------------
# (4) 학습 함수
# --------------------------------------------------------------------------------
def train_model(hidden_size=100, window_size=5, batch_size=32, num_epochs=100, learning_rate=0.001):
    """
    - Mobius에서 훈련 데이터(MOBIUS_BASE_AE_URL) 가져오기
    - 전처리 후 PyTorch DataLoader 생성
    - WeatherLSTM 모델 학습
    - 모델 파라미터 저장(weather_lstm_model.pth)
    """
    print("[INFO] Fetching training data...")
    train_data = fetch_data(MOBIUS_BASE_AE_URL)
    print("[INFO] Preprocessing training data...")
    print(train_data)
    
    print("[INFO] Preprocessing training data...")
    X_train, y_train, scaler, encoder = preprocess_data(train_data, window_size)

    # 텐서 변환 및 DataLoader 생성
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 모델/손실함수/최적화함수 정의
    input_size  = X_train.shape[2]
    output_size = y_train.shape[1]
    print(f"[INFO] Input size: {input_size}, Output size: {output_size}")
    input()
    model = WeatherLSTM(input_size, hidden_size, output_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 학습 루프
    print("[INFO] Starting training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)                       # (batch, output_size)
            loss    = criterion(outputs, torch.argmax(y_batch, dim=1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")

    # 학습 완료 후 모델 저장
    torch.save(model.state_dict(), "weather_lstm_model.pth")
    print("[INFO] Model trained and saved: weather_lstm_model.pth")

# --------------------------------------------------------------------------------
# (5) 테스트 함수
# --------------------------------------------------------------------------------
def test_model(hidden_size=100, window_size=5):
    """
    - Mobius에서 테스트 데이터(MOBIUS_TEST_AE_URL) 가져오기
    - 전처리 후 PyTorch DataLoader 생성
    - 저장된 모델 가중치 로드(weather_lstm_model.pth)
    - 모델 성능 측정(손실, 정확도)
    """
    print("[INFO] Fetching test data...")
    test_data = fetch_data(MOBIUS_TEST_AE_URL)

    print("[INFO] Preprocessing test data...")
    X_test, y_test, scaler, encoder = preprocess_data(test_data, window_size)

    # 텐서 변환 및 DataLoader 생성
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32)
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 모델 준비 (동일 구조)
    input_size  = X_test.shape[2]
    output_size = y_test.shape[1]
    model       = WeatherLSTM(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load("weather_lstm_model.pth"))
    model.eval()

    # 평가
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct    = 0
    total      = 0

    print("[INFO] Starting evaluation...")
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)  
            loss    = criterion(outputs, torch.argmax(y_batch, dim=1))
            total_loss += loss.item()

            predicted = torch.argmax(outputs, dim=1)
            correct  += (predicted == torch.argmax(y_batch, dim=1)).sum().item()
            total    += y_batch.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total
    print(f"[RESULT] Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")

# --------------------------------------------------------------------------------
# 메인 실행부
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    print("[MAIN] Training model...")
    train_model(hidden_size, window_size, batch_size, num_epochs, learning_rate)

    print("[MAIN] Testing model...")
    test_model(hidden_size, window_size)
