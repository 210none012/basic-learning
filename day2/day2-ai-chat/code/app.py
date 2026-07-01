import os
from openai import OpenAI
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# DeepSeek API 配置（使用 OpenAI SDK）
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', 'key')
DEEPSEEK_MODEL = os.environ.get('DEEPSEEK_MODEL', 'deepseek-v4-flash')

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url='https://api.deepseek.com/v1',
)

# 存储每个会话的对话历史
sessions = {}

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Q&A Assistant</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            min-height: 100vh; display: flex; justify-content: center; align-items: center; padding: 20px;
        }
        .chat-container {
            width: 100%; max-width: 800px; height: 90vh;
            background: rgba(255,255,255,0.05); backdrop-filter: blur(20px);
            border-radius: 20px; border: 1px solid rgba(255,255,255,0.1);
            display: flex; flex-direction: column; overflow: hidden;
            box-shadow: 0 25px 50px rgba(0,0,0,0.3);
        }
        .chat-header {
            padding: 20px 24px; border-bottom: 1px solid rgba(255,255,255,0.08);
            display: flex; align-items: center; gap: 12px;
        }
        .chat-header .logo {
            width: 40px; height: 40px; background: linear-gradient(135deg, #667eea, #764ba2);
            border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 20px;
        }
        .chat-header .title { color: #fff; font-size: 18px; font-weight: 600; }
        .chat-header .subtitle { color: rgba(255,255,255,0.5); font-size: 13px; }
        .chat-messages {
            flex: 1; overflow-y: auto; padding: 20px 24px;
            display: flex; flex-direction: column; gap: 16px;
        }
        .chat-messages::-webkit-scrollbar { width: 4px; }
        .chat-messages::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 2px; }
        .message { display: flex; gap: 10px; animation: fadeIn 0.3s ease; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
        .message.user { flex-direction: row-reverse; }
        .message .avatar {
            width: 34px; height: 34px; border-radius: 10px;
            display: flex; align-items: center; justify-content: center; font-size: 16px; flex-shrink: 0;
        }
        .message.user .avatar { background: linear-gradient(135deg, #667eea, #764ba2); }
        .message.assistant .avatar { background: linear-gradient(135deg, #11998e, #38ef7d); }
        .message .bubble {
            max-width: 75%; padding: 12px 16px; border-radius: 16px; line-height: 1.6; font-size: 14px;
        }
        .message.user .bubble {
            background: linear-gradient(135deg, #667eea, #764ba2); color: #fff; border-bottom-right-radius: 4px;
        }
        .message.assistant .bubble {
            background: rgba(255,255,255,0.08); color: rgba(255,255,255,0.9); border-bottom-left-radius: 4px;
        }
        .message.assistant .bubble pre {
            background: rgba(0,0,0,0.3); padding: 12px; border-radius: 8px;
            overflow-x: auto; margin: 8px 0; font-size: 13px;
        }
        .message.assistant .bubble code { background: rgba(0,0,0,0.2); padding: 2px 6px; border-radius: 4px; font-size: 13px; }
        .chat-input-area { padding: 16px 24px; border-top: 1px solid rgba(255,255,255,0.08); display: flex; gap: 10px; }
        .chat-input-area input {
            flex: 1; padding: 12px 18px; background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.1); border-radius: 14px;
            color: #fff; font-size: 14px; outline: none; transition: all 0.3s;
        }
        .chat-input-area input:focus { border-color: rgba(102,126,234,0.5); background: rgba(255,255,255,0.1); }
        .chat-input-area input::placeholder { color: rgba(255,255,255,0.3); }
        .chat-input-area button {
            padding: 12px 20px; background: linear-gradient(135deg, #667eea, #764ba2);
            border: none; border-radius: 14px; color: #fff; font-size: 14px;
            font-weight: 600; cursor: pointer; transition: all 0.3s;
        }
        .chat-input-area button:hover { transform: translateY(-1px); box-shadow: 0 8px 20px rgba(102,126,234,0.3); }
        .chat-input-area button:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
        .typing-indicator { display: flex; gap: 4px; padding: 8px 0; }
        .typing-indicator span {
            width: 6px; height: 6px; background: rgba(255,255,255,0.4);
            border-radius: 50%; animation: bounce 1.4s infinite ease-in-out;
        }
        .typing-indicator span:nth-child(1) { animation-delay: 0s; }
        .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
        .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
        @keyframes bounce { 0%,80%,100% { transform: translateY(0); } 40% { transform: translateY(-6px); } }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <div class="logo">&#x1f916;</div>
            <div>
                <div class="title">AI Q&A Assistant</div>
                <div class="subtitle">Powered by DeepSeek</div>
            </div>
        </div>
        <div class="chat-messages" id="messages">
            <div class="message assistant">
                <div class="avatar">&#x1f916;</div>
                <div class="bubble">Hello! I am an AI Q&A assistant powered by DeepSeek.<br>Feel free to ask me anything!</div>
            </div>
        </div>
        <div class="chat-input-area">
            <input type="text" id="userInput" placeholder="Enter your question..." onkeydown="if(event.key=='Enter') sendMessage()">
            <button id="sendBtn" onclick="sendMessage()">Send</button>
        </div>
    </div>
    <script>
        const messagesDiv = document.getElementById("messages");
        const userInput = document.getElementById("userInput");
        const sendBtn = document.getElementById("sendBtn");
        const sessionId = "session_" + Date.now() + "_" + Math.random().toString(36).substr(2, 9);
        function addMessage(role, content) {
            const d = document.createElement("div");
            d.className = "message " + role;
            d.innerHTML = '<div class="avatar">' + (role === "user" ? "&#x1f464;" : "&#x1f916;") + '</div><div class="bubble">' + escapeHtml(content) + '</div>';
            messagesDiv.appendChild(d);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            return d;
        }
        function addTypingIndicator() {
            const d = document.createElement("div");
            d.className = "message assistant";
            d.id = "typing-indicator";
            d.innerHTML = '<div class="avatar">&#x1f916;</div><div class="bubble"><div class="typing-indicator"><span></span><span></span><span></span></div></div>';
            messagesDiv.appendChild(d);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            return d;
        }
        function removeTypingIndicator() {
            const el = document.getElementById("typing-indicator");
            if (el) el.remove();
        }
        function escapeHtml(text) {
            const d = document.createElement("div");
            d.textContent = text;
            return d.innerHTML;
        }
        async function sendMessage() {
            const msg = userInput.value.trim();
            if (!msg) return;
            userInput.disabled = true;
            sendBtn.disabled = true;
            addMessage("user", msg);
            userInput.value = "";
            addTypingIndicator();
            try {
                const resp = await fetch("/api/chat", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ session_id: sessionId, message: msg })
                });
                const data = await resp.json();
                removeTypingIndicator();
                if (data.error) addMessage("assistant", "Error: " + data.error);
                else addMessage("assistant", data.reply);
            } catch (err) {
                removeTypingIndicator();
                addMessage("assistant", "Network error, please try again");
            }
            userInput.disabled = false;
            sendBtn.disabled = false;
            userInput.focus();
        }
    </script>
</body>
</html>"""



@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data:
        return jsonify({"error": "invalid request"}), 400

    session_id = data.get("session_id", "default")
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "message is empty"}), 400

    if session_id not in sessions:
        sessions[session_id] = []

    sessions[session_id].append({"role": "user", "content": user_message})

    # 调用 DeepSeek API（使用 OpenAI SDK）
    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=sessions[session_id],
            temperature=0.7,
            max_tokens=2000,
        )
        assistant_reply = response.choices[0].message.content

        sessions[session_id].append({"role": "assistant", "content": assistant_reply})

        return jsonify({"reply": assistant_reply})

    except Exception as e:
        return jsonify({"error": f"API error: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
