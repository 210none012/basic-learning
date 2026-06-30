# 使用 Python 官方镜像
FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY app.py .

# 暴露端口
EXPOSE 8000

# 启动应用
CMD ["python", "app.py"]
