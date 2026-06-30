# 实验二：Docker、SSH、Python API 调用与 WSL

## 一、实验目的

1. 学会使用 Docker 构建镜像并部署到远程服务器
2. 学会使用 SSH 连接远程 Linux 服务器
3. 学会在 Python 中调用 API 接口
4. 学会使用 WSL（Windows Subsystem for Linux）

## 二、实验环境

- 操作系统：Windows 11
- Docker：Docker Desktop 29.6.1 (WSL2 后端)
- Python：3.14.6
- 远程服务器：阿里云 ECS (Ubuntu, 121.41.1.96)
- WSL：Ubuntu (WSL2)

## 三、实验项目

编写一个基于 DeepSeek API 的 AI 问答网页程序，用 Docker 打包部署到远程服务器。

## 四、实验步骤

### 步骤 1：编写 Python 后端

使用 Flask 框架编写后端，调用 DeepSeek API 实现 AI 对话。

关键代码 (`app.py`)：
- Flask 提供 Web 服务，前端 HTML 内嵌在模板中
- POST `/api/chat` 接收用户消息，调用 DeepSeek API 返回回复
- 使用会话机制保持多轮对话上下文
- DeepSeek 模型：`deepseek-v4-flash`

### 步骤 2：编写前端页面

内嵌于 Flask 的 HTML/CSS/JS 单页应用：
- 暗色渐变背景，毛玻璃效果容器
- 聊天气泡界面，用户消息和 AI 回复区分左右
- 打字动画加载效果
- 支持 Enter 键发送消息

### 步骤 3：Docker 镜像构建

编写 Dockerfile：
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
EXPOSE 8000
CMD ["python", "app.py"]
```

构建镜像：
```powershell
docker build -t ai-chat:latest day2-ai-chat/code
```
![ping-mu-jie-tu-2026-06-30-163843.png](https://i.postimg.cc/597CHGLP/ping-mu-jie-tu-2026-06-30-163843.png)
遇到的问题：Docker Hub 在国内被墙，拉取基础镜像极慢。
解决方案：配置国内镜像加速源。

### 步骤 4：本地测试

```powershell
# 启动容器
docker run -d -p 8000:8000 --name ai-chat ai-chat:latest

# 浏览器访问
http://127.0.0.1:8000
```
测试结果：
- 前端页面正常渲染
- API 调用 DeepSeek 返回正确回复
- 中文对话正常（JSON_AS_ASCII = False）
![ping-mu-jie-tu-2026-06-30-204731-1.png](https://i.postimg.cc/qBb3tYyW/ping-mu-jie-tu-2026-06-30-204731-1.png)
### 步骤 5：SSH 连接远程服务器

```powershell
ssh root@121.41.1.96
```

服务器信息：
- IP：121.41.1.96
- 系统：Ubuntu
- 实例 ID：iZbp16vieus5shfnl16fhuZ

### 步骤 6：部署到远程服务器

```powershell
# 导出镜像
docker save ai-chat:latest -o ai-chat.tar

# 上传到服务器
scp ai-chat.tar ecs-assist-user@121.41.1.96:~/
```

在服务器上：
```bash
# 加载并运行
docker load -i ~/ai-chat.tar
docker run -d -p 8000:8000 --name ai-chat --restart always ai-chat:latest
```
![ping-mu-jie-tu-2026-06-30-163927.png](https://i.postimg.cc/cCbgVgdd/ping-mu-jie-tu-2026-06-30-163927.png)
![ping-mu-jie-tu-2026-06-30-170719.png](https://i.postimg.cc/mZXHcqC5/ping-mu-jie-tu-2026-06-30-170719.png)
验证：
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"test","message":"你好"}'
```

### 步骤 7：WSL 环境配置

```powershell
# 确认 WSL2 已运行（Docker Desktop 依赖它）
wsl --list --verbose
# 输出：docker-desktop Running, Ubuntu Running

# 安装 Ubuntu 发行版
wsl --install -d Ubuntu

# 进入 Ubuntu
wsl -d Ubuntu
# 设置用户名密码后即可使用
```

WSL 中项目路径：`/mnt/c/Users/16696/Documents/basic-learning/`

## 五、Docker 常用命令及知识总结

| 命令                                            | 说明          |
| --------------------------------------------- | ----------- |
| `docker build -t name:tag .`                  | 构建镜像        |
| `docker images`                               | 查看本地镜像      |
| `docker run -d -p 8000:8000 --name xxx image` | 后台运行容器，映射端口 |
| `docker ps`                                   | 查看运行中的容器    |
| `docker ps -a`                                | 查看所有容器      |
| `docker stop/start/restart name`              | 停止/启动/重启    |
| `docker rm name`                              | 删除容器        |
| `docker rmi image`                            | 删除镜像        |
| `docker logs name`                            | 查看容器日志      |
| `docker exec -it name bash`                   | 进入容器终端      |
| `docker save image -o file.tar`               | 导出镜像        |
| `docker load -i file.tar`                     | 导入镜像        |

## 命令

- docker pull docker.io/library/nginx:latest(registry：仓库地址（官方发布可省略）/namespace：命名空间（作者名）（官方发布可省略）/名称：tag：标签（版本号）)
- docker.io/library/nginx是一个镜像库，存放同一镜像的不同版本
- docker run = docker pull + docker run当镜像不存在时会自动拉取
- docker容器与宿主机是隔离的，因此需要进行端口映射访问（-p 宿主机端口:容器端口）
- 挂载卷：将宿主机与容器内目录绑定，相互影响，防止删除容器时，删除所有数据（-v 宿主机目录:容器目录）绑定后宿主机目录会覆盖容器目录
- sudo docker volume create nginx_html后可直接用nginx_html代替宿主机目录，可使用docker volume inspect nginx_html查看挂载卷名字
- run -it ：让控制台进入容器
- run --rm：当容器停止时删除
- run --restart always（自动重启） /unless-stopped（手动停止的不再重启）
- 每次docker run 都会启动一个新的容器，docker start/stop可以实现对原有容器的启停
- docker create 只创建容器，但不立即启动，
- docker logs 用于查看日志
- docker 使用了Cgroup进行限制和隔离进程的资源使用，可以设置资源上限，避免影响宿主机。Namespaces用于隔离进程的资源视图，使容器只能看到内部的进程ID，网络资源和文件目录。docker内部类似于一个Linux系统。
- 可以使用docker exec ID ps -ef查看容器内进程
- docker exec -it ID /bin/sh可以进入容器内执行Linux命令。

## Dockerfile

- Dockerfile第一行为FROM 基础镜像
- WORKDIR类似于cd，用于切换到容器内目录
- COPY用于拷贝文件
- RUN用于执行命令
- EXPOSE声明镜像提供服务的端口（仅声明）
- CMD为容器自动执行命令，建议写成数组形式。

## 推送

- 推送镜像时需要带上用户名
  docker build -t 用户名/镜像名
  docker push 用户名/镜像名

## Docker网络

- Docker网络默认为bridge（桥接模式），所有容器默认连接该网络，容器之间可以互相访问，但与宿主机隔离。
- docker network create network1可以创建子网，同一子网下容器可互相访问（run --network指定加入子网），子网内使用名字即可互相访问
- Host模式下容器共享宿主机的网络，容器直接使用宿主机的IP地址，服务直接运行在宿主机端口（run --network host)
- none模式：不联网
## 六、SSH 常用命令总结

| 命令 | 说明 |
|------|------|
| `ssh user@host` | 密码/密钥登录 |
| `ssh -p 2222 user@host` | 指定端口登录 |
| `ssh-keygen -t ed25519` | 生成密钥对 |
| `ssh-copy-id user@host` | 复制公钥到服务器 |
| `scp file user@host:path` | 上传文件 |
| `scp user@host:path ./` | 下载文件 |
| `scp -r dir user@host:path` | 上传目录 |

## 七、实验结果

1. Docker 镜像构建成功，国内镜像源加速正常
2. 本地 Docker 容器运行 AI 问答网页，功能完整
3. SSH 成功连接阿里云 ECS 服务器
4. 镜像部署到服务器，API 返回正确
5. WSL2 Ubuntu 环境配置完成

## 八、项目结构

```
day2/
├── node.md                    # 本实验报告
└── day2-ai-chat/              # AI 问答网页项目
    ├── README.md              # 项目说明
    └── code/
        ├── app.py             # Flask 后端 + 前端
        ├── Dockerfile         # Docker 构建文件
        └── requirements.txt   # Python 依赖
```

## 九、实验总结

通过本次实验，掌握了以下技能：

- **Docker 全流程**：编写 Dockerfile → 构建镜像 → 本地运行 → 导出 → 远程部署
- **SSH 远程操作**：SSH 登录、SCP 文件传输、服务器环境配置
- **Python API 调用**：使用 `requests` 库调用 DeepSeek 等第三方 API
- **Flask Web 开发**：搭建 Web 服务、前后端交互
- **WSL 环境**：Windows 上运行 Linux，实现本地与服务器环境一致

### 遇到的问题与解决

| 问题                       | 解决方案                           |
| ------------------------ | ------------------------------ |
| Docker Hub 拉取镜像超时        | 配置国内镜像加速源                      |
| Docker Desktop 修改配置卡死    | 直接编辑 daemon.json 文件            |
| SSH 连接 Permission denied | 阿里云 ECS 需用密钥或重置密码              |

