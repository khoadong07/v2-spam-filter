const { io } = require("socket.io-client");

// Configuration
const NUM_REQUESTS = 1000;               // Maximum number of requests to send
const TEST_DURATION = 60 * 1000;         // Benchmark duration: 60 seconds
const TARGET_URL = "http://148.113.218.245:5001";

let successCount = 0;                    // Counter for successful responses
let totalTime = 0;                       // Total accumulated response time
let startTimes = {};                     // Store start time of each request by job ID
let done = false;                        // Flag to stop sending new requests

// Function to send a single request with a unique job ID
const sendRequest = (socket, i) => {
  const jobId = `job_${i}_${Date.now()}`;
  startTimes[jobId] = Date.now();

  const data = {
    category: "finance",
    data: [
      {
        id: jobId,
        topic: "MCredit",
        topic_id: "123",
        title: "",
        content: "Trong bối cảnh bị mạng xã hội cạnh tranh gay gắt, nhiều chuyên gia cho rằng báo chí cần trở thành nơi để độc giả kiểm chứng thông tin...",
        description: "",
        sentiment: "Neutral",
        site_name: "Threads - blam0_gerard_way_food",
        site_id: "63479744980",
        label: "Minigame/ livestream",
        type: "fbPageComment"
      }
    ]
  };

  // Emit prediction request to server
  socket.emit("predict", data);
};

// Connect to the inference server
const socket = io(TARGET_URL, {
  transports: ["websocket"],
  reconnection: false
});

// Once connected to server
socket.on("connect", () => {
  console.log("✅ Connected to server");

  let i = 0;

  // Start sending requests at regular intervals
  const interval = setInterval(() => {
    if (done || i >= NUM_REQUESTS) {
      clearInterval(interval);
      return;
    }
    sendRequest(socket, i++);
  }, 50); // Send one request every 50ms (~20 requests/sec)

  // After the test duration, stop sending new requests
  setTimeout(() => {
    done = true;
    console.log("⏳ Benchmark finished. Waiting for remaining responses...");

    // Wait additional 10 seconds to receive remaining responses
    setTimeout(() => {
      const avgTime = successCount === 0 ? 0 : totalTime / successCount;
      console.log(`✅ Total requests succeeded: ${successCount}`);
      console.log(`⏱️ Average response time: ${(avgTime / 1000).toFixed(2)} seconds`);
      socket.disconnect();
    }, 10000);
  }, TEST_DURATION);
});

// Handle the result response from server
socket.on("result", (data) => {
  const jobId = data.results?.[0]?.id;
  const duration = Date.now() - (startTimes[jobId] || Date.now());

  totalTime += duration;
  successCount += 1;

  delete startTimes[jobId]; // Clean up start time to free memory
});

// Handle connection errors
socket.on("connect_error", (err) => {
  console.error("⚠️ Connection error:", err.message);
});

// Log disconnection
socket.on("disconnect", () => {
  console.log("❌ Disconnected from server");
});
