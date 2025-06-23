import socketio
import pandas as pd
from langdetect import detect_langs, DetectorFactory

DetectorFactory.seed = 0

sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
app = socketio.ASGIApp(sio)

def detect_language(text: str):
    try:
        detections = detect_langs(text)
        if detections:
            top = detections[0]
            return {"language": top.lang, "confidence": round(top.prob, 4)}
    except:
        return {"language": "unknown", "confidence": 0.0}
    return {"language": "unknown", "confidence": 0.0}

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

@sio.event
async def connect(sid, environ):
    print(f"[Server] Connected: {sid}")

@sio.event
async def disconnect(sid):
    print(f"[Server] Disconnected: {sid}")

@sio.event
async def detect_text(sid, data):
    try:
        result = process_data(data)
        await sio.emit('detect_result', result, to=sid)
    except Exception as e:
        await sio.emit('detect_result', {"error": str(e)}, to=sid)
