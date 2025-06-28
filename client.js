// client.js
const { io } = require("socket.io-client");

const socket = io("http://localhost:5001");

socket.on("connect", () => {
  console.log("✅ Connected to server");

  // Dữ liệu mẫu
  const data = {
    category: "healthcare",
    data: [
      {
        id: "1",
        title: "Nhận ngay 10GB miễn phí",
        content: "Chương trình khuyến mãi đặc biệt từ nhà mạng",
        description: "Click vào link để tham gia ngay!"
      },
      {
        id: "2",
        title: "Thông báo tài khoản",
        content: "Bạn có hóa đơn cần thanh toán",
        description: ""
      }
    ]
  };

  console.log("📤 Sending predict request...");
  socket.emit("predict", data);
});

socket.on("result", (data) => {
  console.log("📥 Received result:");
  console.dir(data, { depth: null });
  socket.disconnect();
});

socket.on("disconnect", () => {
  console.log("❌ Disconnected from server");
});

socket.on("connect_error", (err) => {
  console.error("⚠️ Connection error:", err.message);
});
