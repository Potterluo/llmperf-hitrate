FROM python:3.11-slim

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

# 使用清华源加速（如需离线构建，请提前下载 whl 文件改为 COPY 安装）
RUN pip install --no-cache-dir --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 验证安装
RUN python -c "import pytest, pandas, transformers, pydantic; print('All packages installed successfully')"

ENTRYPOINT [""]