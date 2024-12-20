#!/bin/bash

# 检查是否传递了版本号
if [ -z "$1" ]; then
  echo "Usage: $0 <version>"
  exit 1
fi

# 获取版本号
VERSION=$1

# 执行 Docker 构建命令
docker build -f app/Dockerfile -t interview_eval:$VERSION .

# 输出构建成功信息
if [ $? -eq 0 ]; then
  echo "Docker image 'interview_eval:$VERSION' built successfully!"
else
  echo "Docker build failed!"
  exit 1
fi
