
const { io } = require("socket.io-client");

const socket = io("http://localhost:5001");

socket.on("connect", () => {
  console.log("✅ Connected to server");

  const data = {
    category: "finance",
    data: [
      {
        id: "63479744980_3642136556718832910",
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
