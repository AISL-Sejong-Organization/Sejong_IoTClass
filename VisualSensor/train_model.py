import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

import numpy as np
import requests
from PIL import Image
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------------
# (1) 재현성(Reproducibility) 설정
# --------------------------------------------------------------------------------
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# --------------------------------------------------------------------------------
# (2) 하이퍼파라미터 및 변환 설정
# --------------------------------------------------------------------------------
batch_size     = 32
num_epochs     = 10
learning_rate  = 0.001
output_size    = 10  # MNIST(0~9) 분류이므로 클래스 수=10

# torchvision.Transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # MNIST 흑백 이미지를 평균=0.5, 표준편차=0.5로 정규화
])

# --------------------------------------------------------------------------------
# (3) Mobius 플랫폼 설정
# --------------------------------------------------------------------------------
MOBIUS_BASE_URL       = "http://203.250.148.120:20519/Mobius/"
MOBIUS_TRAIN_AE_NAME  = "testman14"
MOBIUS_TEST_AE_NAME   = "testman15"

# 실제 AE URL
MOBIUS_TRAIN_URL = os.path.join(MOBIUS_BASE_URL, MOBIUS_TRAIN_AE_NAME)
MOBIUS_TEST_URL  = os.path.join(MOBIUS_BASE_URL, MOBIUS_TEST_AE_NAME)

# GET 요청 시 사용될 헤더
HEADERS_GET = {
    'Accept':       'application/json',
    'X-M2M-RI':     '12345',
    'X-M2M-Origin': 'SOrigin'
}

# --------------------------------------------------------------------------------
# (4) 센서(컨테이너) 목록: 0~9 (MNIST)
# --------------------------------------------------------------------------------
CONTAINERS = {str(i): f"{i}" for i in range(10)}  
# 예: {'0': '0', '1': '1', ..., '9': '9'}

# --------------------------------------------------------------------------------
# (5) MNISTCNN 모델 정의
# --------------------------------------------------------------------------------
# CNN Model Definition
class MNISTCNN(nn.Module):
    def __init__(self, output_size=10):
        super(MNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, output_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Base64 모델 디코딩 함수
def decode_model(base64_encoded_str):
    decoded_bytes = base64.b64decode(base64_encoded_str)
    model_path = "./model/mnist_cnn.pth"
    with open(model_path, "wb") as f:
        f.write(decoded_bytes)
    return model_path


# --------------------------------------------------------------------------------
# (6) Mobius 연동: HTTP GET
# --------------------------------------------------------------------------------
def http_get(url, params=None, headers=None, iotPlatform=None):
    """
    주어진 URL에 GET 요청. iotPlatform=True일 경우 OneM2M 기본 헤더 적용.
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
        print(f"[Error] Connection timed out for URL: {url}")
        return None

    except requests.HTTPError as http_err:
        print(f"[Error] HTTP error occurred for URL {url}: {http_err}")
        return None

    except Exception as err:
        print(f"[Error] An error occurred for URL {url}: {err}")
        return None

# --------------------------------------------------------------------------------
# (7) Mobius 연동: 특정 컨테이너 URL 하위의 모든 CIN(con) 가져오기
# --------------------------------------------------------------------------------
def all_cin_get_uri(path, max_retries=10):
    """
    - path + '?fu=1&ty=4' 쿼리로 모든 콘텐츠 인스턴스(CIN) 리소스 URIs 가져옴.
    - 각 URI를 GET -> 'con' 필드만 추출하여 리스트로 반환.
    """
    # 모든 CIN 조회
    path = path + '?fu=1&ty=4'
    parsed_path = urlparse(path)
    base_path   = f"{parsed_path.scheme}://{parsed_path.netloc}/"

    con_list = []
    all_uri  = http_get(path, iotPlatform=True)
    if not all_uri or "m2m:uril" not in all_uri:
        return con_list  # 에러 시 또는 데이터 없을 시 빈 리스트

    for uri in all_uri["m2m:uril"]:
        retries = 0
        while retries < max_retries:
            cin_json = http_get(base_path + uri, iotPlatform=True)
            if cin_json is not None:
                con      = cin_json["m2m:cin"]["con"]  # 실제 데이터
                con_list.append(con)
                break
            else:
                retries += 1
                print(f"[Retry {retries}] for URL: {base_path + uri}")

        if retries == max_retries:
            print(f"[FAIL] Data not fetched after {max_retries} attempts for URL: {base_path + uri}")

    return con_list

# --------------------------------------------------------------------------------
# (8) Mobius에서 (0~9) 컨테이너에 있는 이미지 데이터를 가져오기
# --------------------------------------------------------------------------------
def fetch_data(base_url):
    """
    - base_url 아래 CONTAINERS(0~9) 각각 접근 -> 이미지 JSON 가져오기
    - 리턴: {'images': [...], 'labels': [...]}
    """
    data = {'images': [], 'labels': []}
    for digit, container_name in CONTAINERS.items():
        url = f"{base_url}/{container_name}"
        con_list = all_cin_get_uri(url)
        
        for con in con_list:
            # con = {"image": [[...], ...]} 형태라고 가정
            data['images'].append(con['image'])  # JSON에서 'image' 키 꺼냄
            data['labels'].append(int(digit))    # label은 0~9 정수
    return data

# --------------------------------------------------------------------------------
# (9) PyTorch Dataset 정의
# --------------------------------------------------------------------------------
class MNISTDataset(Dataset):
    """
    Mobius에서 받아온 이미지 배열 + 라벨을
    PyTorch Dataset 형식으로 바꾼다.
    """
    def __init__(self, data, transform=None):
        # data['images']: list of image arrays
        # data['labels']: list of int labels
        self.images    = [np.array(img, dtype=np.float32) for img in data['images']]
        self.labels    = data['labels']
        self.transform = transform

    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Numpy -> PIL
        image = Image.fromarray(image)
        # Transform (ToTensor, Normalize 등)
        if self.transform:
            image = self.transform(image)

        return image, label

# --------------------------------------------------------------------------------
# (10) 학습 함수
# --------------------------------------------------------------------------------
def train_model(model, data_loader, criterion, optimizer, num_epochs=10):
    """
    - model (CNN)
    - data_loader (train)
    - criterion (CrossEntropyLoss)
    - optimizer (Adam, etc.)
    - num_epochs (기본 10)
    """
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for images, labels in data_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # 학습 완료 후 모델 저장
    torch.save(model.state_dict(), "mnist_cnn.pth")
    print("[INFO] Model saved as mnist_cnn.pth")

# --------------------------------------------------------------------------------
# (11) 테스트(검증) 함수
# --------------------------------------------------------------------------------
def test_model(model, data_loader):
    """
    - model
    - data_loader (test set)
    - 정확도(accuracy) 측정
    """
    model.eval()
    correct = 0
    total   = 0

    with torch.no_grad():
        for images, labels in data_loader:
            outputs    = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total     += labels.size(0)
            correct   += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"[RESULT] Test Accuracy: {accuracy:.4f}")

# --------------------------------------------------------------------------------
# (12) 메인 실행부
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    print("[MAIN] Fetching train data from AE:", MOBIUS_TRAIN_URL)
    train_data = fetch_data(MOBIUS_TRAIN_URL)

    print("[MAIN] Fetching test data from AE:", MOBIUS_TEST_URL)
    test_data  = fetch_data(MOBIUS_TEST_URL)

    # PyTorch Dataset / DataLoader
    train_dataset = MNISTDataset(train_data, transform=transform)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset  = MNISTDataset(test_data, transform=transform)
    test_loader   = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 모델, 손실함수, 옵티마이저
    model     = MNISTCNN(output_size=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("[MAIN] Training model...")
    train_model(model, train_loader, criterion, optimizer, num_epochs)

    print("[MAIN] Testing model with saved weights...")
    model_loaded = MNISTCNN(output_size=10)
    model_loaded.load_state_dict(torch.load("mnist_cnn.pth"))
    model_loaded.eval()
    test_model(model_loaded, test_loader)
    print("[MAIN] Done.")
