import socketio
from aiohttp import web
import asyncio
import json
import redis.asyncio as redis
import uuid

# Redis config
REDIS_REQUEST_QUEUE = "spam_request_queue"
REDIS_RESULT_QUEUE = "spam_result_queue"

# Khởi tạo Redis async client
redis_conn = redis.Redis(host="redis", port=6379, decode_responses=True)

# Socket.IO setup
sio = socketio.AsyncServer(async_mode='aiohttp', cors_allowed_origins="*")
app = web.Application()
sio.attach(app)

# Đẩy request vào Redis queue
async def enqueue_request(text, meta):
    job_id = str(uuid.uuid4())
    payload = {
        "job_id": job_id,
        "text": text,
        "meta": meta
    }
    await redis_conn.rpush(REDIS_REQUEST_QUEUE, json.dumps(payload))
    return job_id

# Đợi kết quả từ Redis
async def wait_for_result(job_id, timeout=5):
    for _ in range(timeout * 10):  # Kiểm tra mỗi 100ms
        results = await redis_conn.lrange(REDIS_RESULT_QUEUE, 0, -1)
        for item in results:
            obj = json.loads(item)
            if obj.get("job_id") == job_id:
                await redis_conn.lrem(REDIS_RESULT_QUEUE, 1, item)
                return obj["result"]
        await asyncio.sleep(0.1)
    return {"error": "Timeout"}

# Socket events
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
            "topic_id": item.get("topic_id", ""),
            "title": item.get("title", ""),
            "content": item.get("content", ""),
            "description": item.get("description", ""),
            "sentiment": item.get("sentiment", ""),
            "site_name": item.get("site_name", ""),
            "site_id": item.get("site_id", ""),
            "type": item.get("type", ""),
            "label": item.get("label", ""),
            "category": category
        })

        result = await wait_for_result(job_id)
        result.update({
            "id": item.get("id"),
            "topic": item.get("topic", "")
        })
        results.append(result)

    await sio.emit("result", {"category": category, "results": results}, to=sid)

# Đóng Redis connection khi tắt server
async def close_redis(app):
    await redis_conn.close()

app.on_shutdown.append(close_redis)

# Run the app
if __name__ == '__main__':
    web.run_app(app, port=5001)