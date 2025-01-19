import socket
import os
import time
import base64
import json

# --------------------------------------------------------------------------------
# (1) 로컬 이미지 경로 가져오기
# --------------------------------------------------------------------------------
def get_local_images(data_folder="./data"):
    """
    - data_folder 내 PNG/JPG/JPEG 파일 목록을 반환
    """
    return [
        os.path.join(data_folder, file)
        for file in os.listdir(data_folder)
        if file.endswith((".png", ".jpg", ".jpeg"))
    ]

# --------------------------------------------------------------------------------
# (2) 이미지 → Base64 인코딩
# --------------------------------------------------------------------------------
def image_to_base64(img_path):
    """
    - 로컬 이미지 파일을 읽어서 base64 문자열로 변환
    """
    with open(img_path, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode('utf-8')

# --------------------------------------------------------------------------------
# (3) 메인 실행부: 소켓을 통해 모델 서버에 이미지 전송
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    # 1) 소켓 서버 정보
    host = '127.0.0.1'
    port = 5000

    # 2) 소켓 클라이언트 생성, 서버 연결
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    print(f"[SENSOR] Connected to model server at {host}:{port}")

    # 3) 로컬의 ./data 폴더 이미지 목록
    image_paths = get_local_images("./data")
    print(f"[SENSOR] Found {len(image_paths)} images in ./data")

    # 4) 이미지 하나씩 서버로 전송
    for image_path in image_paths:
        b64_str = image_to_base64(image_path)

        # 전송할 JSON 구성
        msg = {
            "image_data": b64_str
        }
        msg_str = json.dumps(msg)

        # 서버로 전송
        s.sendall(msg_str.encode('utf-8'))
        
        # 서버 응답(추론 결과) 받기
        data = s.recv(4096).decode('utf-8')
        if data:
            result = json.loads(data)
            print(f"[SENSOR] Received inference result: {result}")
        else:
            print("[SENSOR] No response received.")

        # 테스트 목적: 잠시 대기
        time.sleep(1)

    # 5) 전송 완료 후 소켓 종료
    s.close()
    print("[SENSOR] Disconnected from server.")
