import asyncio
import socketio
import time
import statistics

SERVER_URL = "http://localhost:5001"
CONCURRENT_CLIENTS = 50  # Số client đồng thời (test từ 10 -> 500 tùy máy)
REQUEST_PER_CLIENT = 1   # Mỗi client gửi bao nhiêu request

# Mỗi sample request
def generate_data(i):
    return {
        "category": "telecom",
        "data": [
            {
                "id": f"req-{i}-{j}",
                "title": "Khuyến mãi cực sốc",
                "content": "Bạn đã nhận được phần thưởng miễn phí từ nhà mạng",
                "description": "Click vào đường dẫn bên dưới để nhận"
            }
            for j in range(3)
        ]
    }

async def run_client(index, results):
    sio = socketio.AsyncClient()

    start = time.time()

    @sio.on("result")
    async def on_result(data):
        end = time.time()
        duration = end - start
        results.append(duration)
        await sio.disconnect()

    try:
        await sio.connect(SERVER_URL)
        await sio.emit("predict", generate_data(index))
        await sio.wait()
    except Exception as e:
        print(f"[Client {index}] Error:", e)

async def main():
    results = []
    tasks = [run_client(i, results) for i in range(CONCURRENT_CLIENTS)]
    print(f"🚀 Running {CONCURRENT_CLIENTS} concurrent clients...")
    start_time = time.time()
    await asyncio.gather(*tasks)
    total_time = time.time() - start_time

    print("\n📊 Benchmark Results:")
    print(f"- Total clients: {CONCURRENT_CLIENTS}")
    print(f"- Total time   : {total_time:.2f} seconds")
    print(f"- Avg latency  : {statistics.mean(results):.3f} s")
    print(f"- 95% latency  : {statistics.quantiles(results, n=100)[94]:.3f} s")
    print(f"- Max latency  : {max(results):.3f} s")
    print(f"- Throughput   : {CONCURRENT_CLIENTS / total_time:.2f} req/s")

if __name__ == "__main__":
    asyncio.run(main())
