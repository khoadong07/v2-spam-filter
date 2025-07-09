const { io } = require("socket.io-client");

const TARGET_URL = "http://148.113.218.248:5001"; // Thay bằng địa chỉ thật nếu khác

// Tạo jobId ngẫu nhiên
const jobId = `job_${Date.now()}`;

// Tạo dữ liệu test
const testData = {
  category: "retail",
  data: [
    {
      id: "12321321321321",
      topic: "MCredit",
      topic_id: "123",
      title: "",
      content:
        "Trong bối cảnh bị mạng xã hội cạnh tranh gay gắt, nhiều chuyên gia cho rằng báo chí cần trở thành nơi để độc giả kiểm chứng thông tin, cần nâng cao chất lượng. Tại diễn đàn Báo chí Việt Nam trong kỷ nguyên mới: Tầm nhìn kiến tạo không gian phát triển chiều 19/6, Thứ trưởng Văn hóa Thể thao và Du lịch Lê Hải Bình đánh giá báo chí thế giới đã vận động qua nhiều giai đoạn, sang thiên niên kỷ này đã phát triển rất nhanh,",
      description: "",
      sentiment: "Neutral",
      site_name: "Threads - blam0_gerard_way_food",
      site_id: "63479744980",
      label: "Minigame/ livestream",
      type: "fbPageComment"
    }
  ]
};

// Kết nối tới socket server
const socket = io(TARGET_URL, {
  transports: ["websocket"],
  reconnection: false,
});

// Khi kết nối thành công
socket.on("connect", () => {
  console.log("✅ Connected to server");

  console.log("📤 Sending test request...");
  socket.emit("predict", testData);
});

// Khi nhận được kết quả
socket.on("result", (data) => {
  console.log("📥 Received result:");
  console.dir(data, { depth: null });

  socket.disconnect();
});

// Khi có lỗi kết nối
socket.on("connect_error", (err) => {
  console.error("❌ Connection error:", err.message);
});

// Khi bị ngắt kết nối
socket.on("disconnect", () => {
  console.log("🔌 Disconnected from server");
});
