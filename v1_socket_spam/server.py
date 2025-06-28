# server.py
import socketio
from aiohttp import web
import asyncio
import json
import redis
import uuid

# Redis config
redis_conn = redis.Redis(host="localhost", port=6379, db=0)
REDIS_REQUEST_QUEUE = "spam_request_queue"
REDIS_RESULT_QUEUE = "spam_result_queue"

# Socket.IO setup
sio = socketio.AsyncServer(async_mode='aiohttp', cors_allowed_origins="*")
app = web.Application()
sio.attach(app)

# Đẩy request vào hàng đợi Redis
async def enqueue_request(text, meta):
    job_id = str(uuid.uuid4())
    payload = {
        "job_id": job_id,
        "text": text,
        "meta": meta
    }
    redis_conn.rpush(REDIS_REQUEST_QUEUE, json.dumps(payload))
    return job_id

# Lấy kết quả từ Redis queue
async def wait_for_result(job_id, timeout=5):
    for _ in range(timeout * 10):  # 100ms * 50 = 5s
        results = redis_conn.lrange(REDIS_RESULT_QUEUE, 0, -1)
        for item in results:
            obj = json.loads(item)
            if obj.get("job_id") == job_id:
                redis_conn.lrem(REDIS_RESULT_QUEUE, 1, item)
                return obj["result"]
        await asyncio.sleep(0.1)
    return {"error": "Timeout"}

@sio.event
async def connect(sid, environ):
    print(f"✅ Client {sid} connected")

@sio.event
async def disconnect(sid):
    print(f"❌ Client {sid} disconnected")

@sio.event
async def predict(sid, data):
    category = data.get("category", "")
    items = data.get("data", [])
    results = []

    for item in items:
        text = " ".join([
            item.get("title", ""),
            item.get("content", ""),
            item.get("description", "")
        ]).strip()

        if not text:
            results.append({
                "id": item.get("id"),
                "error": "Empty text"
            })
            continue

        job_id = await enqueue_request(text, {
            "id": item.get("id"),
            "topic": item.get("topic", ""),
            "category": category
        })

        result = await wait_for_result(job_id)
        result.update({
            "id": item.get("id"),
            "topic": item.get("topic", "")
        })
        results.append(result)

    await sio.emit("result", {"category": category, "results": results}, to=sid)

if __name__ == '__main__':
    web.run_app(app, port=5001)
