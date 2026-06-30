# Day3 - AI 问答网页程序

基于 DeepSeek API 的 AI 问答网页应用，支持多轮对话。

## 项目结构

```
day2-ai-chat/
├── README.md              # 项目说明文档
└── code/
    ├── app.py             # Python Flask 后端（含内嵌前端 HTML）
    ├── requirements.txt   # Python 依赖
    └── Dockerfile         # Docker 镜像构建文件
```

## 技术栈

- **后端**：Python Flask
- **前端**：原生 HTML/CSS/JS（内嵌于 Flask 模板）
- **API**：DeepSeek API (deepseek-v4-flash)
- **容器化**：Docker

## 功能特性

- 基于 DeepSeek 大模型的智能问答
- 多轮对话，保持上下文记忆
- 现代化暗色 UI，带打字动画
- Docker 一键部署

## 快速开始

### 方式一：本地运行

```bash
# 1. 安装依赖
cd code
pip install -r requirements.txt

# 2. 启动服务
python app.py

# 3. 打开浏览器访问
# http://127.0.0.1:8000
```

### 方式二：Docker 运行

```bash
# 1. 构建镜像
cd code
docker build -t ai-chat .

# 2. 运行容器
docker run -d -p 8000:8000 --name ai-chat ai-chat

# 3. 打开浏览器访问
# http://127.0.0.1:8000

# 停止和清理
docker stop ai-chat
docker rm ai-chat
```
![](https://i.postimg.cc/597CHGLP/ping-mu-jie-tu-2026-06-30-163843.png)
### 方式三：部署到远程服务器

```bash
# 1. SSH 连接服务器
ssh root@121.41.1.96

# 2. 在服务器上拉取/构建镜像后运行
docker load -i ~/ai-chat.tar
docker run -d -p 8000:8000 --name ai-chat --restart always ai-chat:latest

# 3. 通过浏览器访问
# http://121.41.1.96:8000
```
![[屏幕截图 2026-06-30 170719.png]]
## 环境变量

| 变量名              | 说明              | 默认值               |
| ---------------- | --------------- | ----------------- |
| DEEPSEEK_API_KEY | DeepSeek API 密钥 | 内置默认值             |
| DEEPSEEK_MODEL   | 使用的模型           | deepseek-v4-flash |
| PORT             | 服务端口            | 8000              |

## API 接口

### POST /api/chat

发送消息并获取 AI 回复。

**请求体：**
```json
{
  "session_id": "session_xxx",
  "message": "你好，请介绍一下自己"
}
```

**响应：**
```json
{
  "reply": "你好！我是 DeepSeek 驱动的 AI 助手..."
}
```
![[屏幕截图 2026-06-30 204731 1.png]]