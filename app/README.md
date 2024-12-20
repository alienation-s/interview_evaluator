# Interview Evaluator 项目

本项目使用 Docker 构建和运行 FastAPI 应用。你可以使用提供的 `build.sh` 脚本来构建 Docker 镜像。

## 前置条件

确保你已经安装了以下工具：

- [Docker](https://www.docker.com/get-started)

## 如何构建 Docker 镜像

1. 克隆项目（如果还没有克隆）：

   ```bash
   git clone <仓库地址>
   cd interview_evaluator
   ```
2. 运行 `build.sh` 脚本，并传入你希望为 Docker 镜像打上的版本号。版本号将作为镜像标签使用。

   例如，构建版本为 `v1.0` 的镜像：

```bash
bash app/build.sh v1.0
```

3. 如果构建成功，你将看到类似如下的输出：

```bash
Docker 镜像 'interview_eval:v1.0' 构建成功！
```

4. 你可以使用以下命令来运行容器，并将容器的 7890 端口映射到本地机器的 7890 端口：

```bash
docker run -it -p 7890:7890 --gpus all --name interview_eval_container interview_eval:v1.0
```
