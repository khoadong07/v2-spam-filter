import socketio
import pandas as pd
from langdetect import detect_langs, DetectorFactory
import jwt
import time
import os

# Cấu hình
DetectorFactory.seed = 0
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "") 
ALLOWED_ORIGINS = ['*']
MAX_TEXT_LENGTH = 2000
RATE_LIMIT_SECONDS = 1.0

# Tạo server Socket.IO
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins=ALLOWED_ORIGINS
)
app = socketio.ASGIApp(sio)
last_request_time = {}

# Hàm detect ngôn ngữ
def detect_language(text: str):
    if len(text) > MAX_TEXT_LENGTH:
        return {"language": "text_too_long", "confidence": 0.0}
    try:
        detections = detect_langs(text)
        if detections:
            top = detections[0]
            return {"language": top.lang, "confidence": round(top.prob, 4)}
    except Exception:
        return {"language": "unknown", "confidence": 0.0}
    return {"language": "unknown", "confidence": 0.0}

# Xử lý dữ liệu batch
def process_data(data):
    df = pd.DataFrame(data)
    df = df.rename(columns=lambda x: x.lower())
    df['combined_text'] = df[['title', 'content', 'description']].fillna('').agg(' '.join, axis=1)

    results = []
    for row in df.itertuples():
        detect_result = detect_language(row.combined_text)
        results.append({
            "id": row.id,
            "combined_text": row.combined_text,
            **detect_result
        })
    return results

# Xác thực token từ query
def verify_token_from_environ(environ):
    query_string = environ.get('QUERY_STRING', '')
    params = dict(item.split('=') for item in query_string.split('&') if '=' in item)
    token = params.get('token')

    if not token:
        raise ConnectionRefusedError("Missing token")

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise ConnectionRefusedError("Token expired")
    except jwt.InvalidTokenError:
        raise ConnectionRefusedError("Invalid token")

# Sự kiện connect
@sio.event
async def connect(sid, environ):
    try:
        user = verify_token_from_environ(environ)
        print(f"[Server] Connected: {sid} (user_id: {user.get('user_id')})")
    except ConnectionRefusedError as e:
        print(f"[Server] Unauthorized connection from {sid}: {e}")
        raise e

# Sự kiện disconnect
@sio.event
async def disconnect(sid):
    print(f"[Server] Disconnected: {sid}")
    last_request_time.pop(sid, None)

# Sự kiện xử lý text
@sio.event
async def detect_text(sid, data):
    now = time.time()

    if sid in last_request_time and (now - last_request_time[sid] < RATE_LIMIT_SECONDS):
        await sio.emit('detect_result', {"error": "Rate limit exceeded"}, to=sid)
        return
    last_request_time[sid] = now

    try:
        result = process_data(data)
        await sio.emit('detect_result', result, to=sid)
    except Exception as e:
        print(f"[Server] Error processing data from {sid}: {e}")
        await sio.emit('detect_result', {"error": "Internal server error"}, to=sid)
