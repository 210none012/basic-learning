# 实验二：AI 工具、Docker 与 Linux 基础

## 第一部分：AI 编程工具

### 一、实验目的

1. 学会使用 Codex CLI 进行 AI 辅助编程
2. 了解 Codex 的基本功能和工作方式

### 二、实验环境

- 操作系统：Windows
- 工具：Codex CLI（已安装）
- IDE：VS Code + Codex 插件

### 三、Codex CLI 基本使用

#### 3.1 Codex 是什么

Codex 是 OpenAI 推出的终端级 AI 编程助手，可以直接在命令行中通过自然语言与 AI 交互，完成代码编写、文件操作、Git 操作等任务。

#### 3.2 已验证的功能

通过 Day1 的实验，已验证 Codex 可以完成以下操作：

| 功能 | 说明 | 验证状态 |
|------|------|----------|
| Git 操作 | 添加远程仓库、fetch、checkout、push | ✔ 已验证 |
| 文件读写 | 创建文件、写入内容 | ✔ 已验证 |
| 命令执行 | 运行 shell 命令并解析输出 | ✔ 已验证 |
| 环境检查 | 检查系统安装的工具和版本 | ✔ 已验证 |
| 代码生成 | 根据需求生成代码和文档 | ✔ 已验证 |

#### 3.3 Codex 的工作模式

- **计划模式**：先制定步骤计划，再逐步执行
- **默认模式**：直接执行任务，边做边反馈
- **审批机制**：涉及网络、外部写入等操作需要用户确认

#### 3.4 VS Code Codex 插件配置

在 VS Code 中安装 Codex 插件：

1. 打开 VS Code → 扩展 (Ctrl+Shift+X)
2. 搜索 "Codex"
3. 安装 OpenAI Codex 插件
4. 登录 OpenAI 账号完成认证
5. 在 VS Code 终端中即可使用 Codex CLI

---

## 第二部分：Docker

### 一、实验目的

1. 了解 Docker 容器技术的基本概念
2. 掌握 Docker 的安装和环境配置
3. 学会启动和管理 Docker 容器

### 二、实验原理

#### 2.1 Docker 是什么

Docker 是一个**容器化平台**，可以将应用及其依赖打包到一个轻量级、可移植的容器中，在任何支持 Docker 的系统上运行。

#### 2.2 核心概念

| 概念 | 说明 |
|------|------|
| 镜像 (Image) | 容器的模板，包含应用和运行环境 |
| 容器 (Container) | 镜像的运行实例，相互隔离 |
| Dockerfile | 定义镜像构建步骤的文本文件 |
| Docker Hub | 官方的镜像仓库 |
| 数据卷 (Volume) | 持久化存储，容器间共享数据 |

#### 2.3 与传统虚拟机的区别

| | Docker 容器 | 虚拟机 |
|------|-------------|--------|
| 启动速度 | 秒级 | 分钟级 |
| 资源占用 | MB 级 | GB 级 |
| 隔离级别 | 进程级 | 硬件级 |
| 操作系统 | 共享宿主机内核 | 每个 VM 有独立 OS |

### 三、实验步骤（待执行）

#### 步骤 1：安装 Docker Desktop

`powershell
# Windows 下推荐安装 Docker Desktop
# 下载地址：https://www.docker.com/products/docker-desktop/
`

安装注意事项：
- 需要开启 Hyper-V 或 WSL2 后端
- 建议使用 WSL2 作为 Docker 引擎后端

#### 步骤 2：验证安装

`ash
docker --version          # 查看版本
docker run hello-world    # 运行测试镜像
`

#### 步骤 3：常用 Docker 命令

`ash
# 镜像管理
docker images                    # 列出本地镜像
docker pull <image>              # 拉取镜像
docker rmi <image>               # 删除镜像

# 容器管理
docker run -d -p 8080:80 nginx   # 后台运行 nginx，映射端口
docker ps                        # 查看运行中的容器
docker ps -a                     # 查看所有容器（含已停止）
docker stop <container>          # 停止容器
docker start <container>         # 启动已停止的容器
docker rm <container>            # 删除容器
docker exec -it <container> bash # 进入容器交互终端

# 日志与信息
docker logs <container>          # 查看容器日志
docker inspect <container>       # 查看容器详细信息
`

#### 步骤 4：编写 Dockerfile

`dockerfile
# 示例：一个简单的 Node.js 应用
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
EXPOSE 3000
CMD ["node", "index.js"]
`

#### 步骤 5：Docker Compose

`yaml
# docker-compose.yml 示例
version: '3.8'
services:
  web:
    build: .
    ports:
      - "3000:3000"
  db:
    image: postgres:15
    environment:
      POSTGRES_PASSWORD: example
`

---

## 第三部分：Linux 命令与 WSL

### 一、实验目的

1. 配置 Windows Subsystem for Linux (WSL)
2. 掌握基本的 Linux 命令行操作
3. 学会使用 SSH 连接远程服务器

### 二、实验环境检查

当前状态：
- WSL 子系统：已安装（需安装 Linux 发行版）
- 当前无已安装的 Linux 发行版

### 三、实验步骤

#### 步骤 1：安装 WSL Linux 发行版

`powershell
# 查看可用的 Linux 发行版
wsl --list --online

# 安装 Ubuntu（推荐）
wsl --install -d Ubuntu

# 或安装 Debian
wsl --install -d Debian
`

安装完成后会提示创建用户名和密码。

#### 步骤 2：WSL 基本操作

`powershell
wsl                        # 进入默认 Linux 环境
wsl -d Ubuntu              # 进入指定发行版
wsl --list --verbose       # 查看已安装的发行版
wsl --shutdown             # 关闭所有 WSL 实例
exit                       # 退出 WSL 回到 Windows
`

#### 步骤 3：Linux 基础命令

##### 3.1 文件和目录操作

`ash
pwd                  # 显示当前目录路径
ls                   # 列出目录内容
ls -la               # 列出详细信息（含隐藏文件）
cd <dir>             # 切换目录
mkdir <dir>          # 创建目录
touch <file>         # 创建空文件
cp <src> <dst>       # 复制文件/目录
mv <src> <dst>       # 移动/重命名
rm <file>            # 删除文件
rm -r <dir>          # 递归删除目录
cat <file>           # 查看文件内容
head/tail <file>     # 查看文件开头/结尾
find . -name "*.txt" # 查找文件
`

##### 3.2 系统信息

`ash
uname -a             # 系统信息
df -h                # 磁盘使用情况
free -h              # 内存使用情况
top / htop           # 进程监控
ps aux               # 查看所有进程
whoami               # 当前用户
id                   # 用户和组信息
`

##### 3.3 权限管理

`ash
chmod 755 file       # 修改文件权限
chmod +x script.sh   # 添加执行权限
chown user:group file# 修改文件所有者
sudo <command>       # 以管理员身份执行
`

##### 3.4 文本处理

`ash
grep "pattern" file  # 搜索文本
grep -r "pattern" .  # 递归搜索目录
wc -l file           # 统计行数
sort file            # 排序
uniq                 # 去重
管道符 |              # 连接多个命令
`

##### 3.5 网络操作

`ash
ping <host>          # 测试网络连通性
curl <url>           # 发送 HTTP 请求
wget <url>           # 下载文件
netstat -tlnp        # 查看端口监听
ip addr / ifconfig   # 查看网络接口信息
`

#### 步骤 4：SSH 连接远程服务器

##### 4.1 SSH 是什么

SSH (Secure Shell) 是一种加密的网络协议，用于安全地远程登录和管理服务器。

##### 4.2 基本连接

`ash
# 密码登录
ssh username@hostname
ssh username@192.168.1.100
ssh -p 2222 username@hostname    # 指定端口

# 退出远程连接
exit
`

##### 4.3 SSH 密钥配置（免密登录）

`ash
# 1. 生成 SSH 密钥对
ssh-keygen -t ed25519 -C "your_email@example.com"

# 2. 将公钥复制到服务器
ssh-copy-id username@hostname

# 3. 之后即可免密登录
ssh username@hostname
`

##### 4.4 SSH 配置文件

`ash
# ~/.ssh/config 示例
Host myserver
    HostName 192.168.1.100
    User root
    Port 22
    IdentityFile ~/.ssh/id_ed25519

# 配置后可以直接用别名连接
ssh myserver
`

##### 4.5 SCP 文件传输

`ash
# 上传文件到服务器
scp localfile.txt user@host:/remote/path/

# 从服务器下载文件
scp user@host:/remote/file.txt ./

# 递归传输目录
scp -r localdir user@host:/remote/path/
`

### 四、常用 Linux 命令速查表

| 分类 | 命令 | 说明 |
|------|------|------|
| 导航 | pwd ls cd | 路径、列表、切换 |
| 文件 | 	ouch mkdir cp mv m | 创建、复制、移动、删除 |
| 查看 | cat less head 	ail | 查看文件内容 |
| 权限 | chmod chown sudo | 权限管理 |
| 搜索 | grep ind locate | 搜索文件和内容 |
| 系统 | ps 	op df ree uname | 系统信息 |
| 网络 | ping curl wget ssh scp | 网络操作 |
| 管道 | \| > >> < | 输入输出重定向 |
| 进程 | kill jobs g g | 进程管理 |
| 包管理 | pt install/update (Ubuntu) | 软件安装 |

---

## 四、Day2 实验总结

### AI 工具
- Codex CLI 已可通过命令行完成 Git 操作、文件管理、代码生成等任务
- 支持计划模式和默认模式，带有审批机制保障安全性
- VS Code 插件可进一步提升开发体验

### Docker
- Docker 是轻量级容器化平台，启动快、资源占用少
- 核心流程：编写 Dockerfile → 构建镜像 → 运行容器
- Docker Compose 可编排多容器应用
- 待完成：安装 Docker Desktop

### Linux 命令
- WSL 是 Windows 上运行 Linux 的最佳方式
- 待完成：安装 Ubuntu 发行版
- 掌握了文件操作、权限管理、文本处理、网络操作等基础命令
- SSH 是远程服务器管理的核心工具，支持密码和密钥两种认证方式

### 待办事项
- [ ] 安装 Docker Desktop
- [ ] 安装 WSL Ubuntu 发行版
- [ ] 实践 Docker 容器启动和管理
- [ ] 在 WSL 中练习 Linux 基础命令
- [ ] 配置 SSH 密钥并尝试连接远程服务器
