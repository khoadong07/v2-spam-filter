const { io } = require("socket.io-client");
const TARGET_URL = "http://0.0.0.0:5001"; // Thay bằng địa chỉ server thực tế

// Tạo dữ liệu test với nhiều item
const generateTestData = (numItems) => ({
  category: "healthcare_insurance",
  data: Array.from({ length: numItems }, (_, i) => ({
    id: `test_${i}_${Date.now()}`,
    topic: "MCredit",
    topic_id: `123_${i}`,
    title: `Test Title ${i}`,
    content: `Trong bối cảnh bị mạng xã hội cạnh tranh gay gắt, nhiều chuyên gia cho rằng báo chí cần trở thành nơi để độc giả kiểm chứng thông tin. Nội dung test ${i}.`,
    description: `Description for test ${i}`,
    sentiment: "Neutral",
    site_name: "Threads - test",
    site_id: `63479744980_${i}`,
    label: "Minigame/livestream",
    type: "fbPageComment",
  })),
});

// Hàm đo thời gian phản hồi
const measureLatency = () => {
  const start = Date.now();
  return () => Date.now() - start;
};

// Hàm test một client
const testSingleClient = async (numItems) => {
  return new Promise((resolve, reject) => {
    const socket = io(TARGET_URL, {
      transports: ["websocket"],
      reconnection: false,
      timeout: 60000, // Timeout 60s
    });

    const measure = measureLatency();

    socket.on("connect", () => {
      console.log(`✅ Client connected`);
      socket.emit("predict", generateTestData(numItems));
    });

    socket.on("result", (data) => {
      const latency = measure();
      console.log(`📥 Received result in ${latency}ms`);
      console.dir(data, { depth: null });
      socket.disconnect();
      resolve({ latency, result: data });
    });

    socket.on("connect_error", (err) => {
      console.error(`❌ Connection error: ${err.message}`);
      reject(err);
    });

    socket.on("disconnect", () => {
      console.log(`🔌 Client disconnected`);
    });
  });
};

// Hàm test nhiều client đồng thời
const testConcurrentClients = async (numClients, numItemsPerClient) => {
  console.log(`🚀 Testing ${numClients} concurrent clients with ${numItemsPerClient} items each...`);
  const results = [];
  const start = Date.now();

  const promises = Array.from({ length: numClients }, () =>
    testSingleClient(numItemsPerClient)
  );

  try {
    const responses = await Promise.allSettled(promises);
    responses.forEach((response, i) => {
      if (response.status === "fulfilled") {
        results.push({
          client: i + 1,
          latency: response.value.latency,
          success: true,
          result: response.value.result,
        });
      } else {
        results.push({
          client: i + 1,
          latency: null,
          success: false,
          error: response.reason.message,
        });
      }
    });

    const totalTime = Date.now() - start;
    const successCount = results.filter((r) => r.success).length;
    const avgLatency =
      results
        .filter((r) => r.success)
        .reduce((sum, r) => sum + r.latency, 0) / successCount || 0;

    console.log("\n📊 Test Summary:");
    console.log(`Total time: ${totalTime}ms`);
    console.log(`Successful clients: ${successCount}/${numClients}`);
    console.log(`Average latency: ${avgLatency.toFixed(2)}ms`);
    console.log(`Error rate: ${((numClients - successCount) / numClients * 100).toFixed(2)}%`);

    return results;
  } catch (err) {
    console.error(`🔥 Test failed: ${err.message}`);
    return [];
  }
};

// Chạy test
(async () => {
  // Test với 10 client, mỗi client gửi 5 item
  await testConcurrentClients(100, 5);

  // Test với tải lớn hơn: 50 client, mỗi client gửi 10 item
  // await testConcurrentClients(50, 10);
})();