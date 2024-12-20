# 第一阶段：构建应用
FROM python:3.9-slim-buster
# 设置工作目录
WORKDIR /app
# 复制 requirements.txt 并安装依赖
COPY app/requirements.txt /app/
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --upgrade pip && \
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/ --no-cache-dir --prefer-binary
# 复制 libs/sparkai 文件夹
COPY libs/sparkai /usr/local/lib/python3.9/site-packages/sparkai
# 复制整个项目文件
COPY . /app/
# 暴露 FastAPI 默认端口
EXPOSE 7890
# 启动 FastAPI 应用
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7890", "--reload"]
