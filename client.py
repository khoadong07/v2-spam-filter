import socketio
import json
import time
import random
import uuid

sio = socketio.Client()

@sio.event
def connect():
    print("[Client] Connected to server")

@sio.event
def disconnect():
    print("[Client] Disconnected from server")

@sio.on('detect_result')
def on_detect_result(data):
    print("[Client] Result received:")
    print(json.dumps(data, indent=2, ensure_ascii=False))


def send_detect_request(text_id, title, content, description):
    data = [{
        "id": text_id,
        "topic_id": "auto_test",
        "topic": "StressTest",
        "title": title,
        "content": content,
        "description": description,
        "sentiment": "null",
        "site_name": "TestClient",
        "site_id": "000",
        "label": "null",
        "type": "testLoop"
    }]
    sio.emit("detect_text", data)


# ✅ Kết nối tới server (cập nhật đúng host + port)
try:
    sio.connect('http://149.56.28.93:8988')  # hoặc IP thật nếu không dùng từ localhost
except Exception as e:
    print("[Client] Connection failed:", e)
    exit(1)

# Danh sách mẫu để test ngôn ngữ khác nhau
samples = [
    ("Xin chào, đây là tiếng Việt", "Nội dung tiếng Việt", "Mô tả tiếng Việt"),
    ("Hello, this is English", "English content", "Some description"),
    ("Bonjour, ceci est un test", "Contenu français", "Description en français"),
    ("Hola, este es un texto en español", "Contenido español", "Descripción"),
    ("こんにちは、これは日本語のテストです", "日本語のコンテンツ", "日本語の説明")
]

# Vòng lặp liên tục để test
i = 1
while True:
    title, content, description = random.choice(samples)
    unique_id = str(uuid.uuid4())  # tạo id duy nhất
    send_detect_request(unique_id, title, content, description)
    print(f"[Client] Sent request #{i} - ID: {unique_id}")
    i += 1
    time.sleep(0.5)  # giảm xuống 0.1 nếu muốn stress hơn
