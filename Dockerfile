# 使用 docker.1ms.run 镜像加速；若在内网已可直连 docker.io，
# 可将前缀去掉改为 FROM python:3.11-slim
ARG REGISTRY_MIRROR=docker.1ms.run
FROM ${REGISTRY_MIRROR}/python:3.11-slim

# transformers version:
#   - 4.57.6 (default): for most models (Qwen, Llama, DeepSeek, ...)
#   - 5.2.0  : for GLM-5 series (requires transformers >= 5.0)
# Override at build time: --build-arg TRANSFORMERS_VERSION=5.2.0
ARG TRANSFORMERS_VERSION=4.57.6

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

# 安装依赖：transformers 版本由构建参数控制
# 使用清华源加速（如需离线构建，请提前下载 whl 文件改为 COPY 安装）
RUN pip install --no-cache-dir --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir "transformers==${TRANSFORMERS_VERSION}" -i https://pypi.tuna.tsinghua.edu.cn/simple

# 验证安装
RUN python -c "import pytest, pandas, transformers, pydantic, requests, httpx; print('transformers', transformers.__version__, '- all packages OK')"

ENTRYPOINT [""]
