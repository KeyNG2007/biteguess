FROM ubuntu:22.04
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 APP_DATA_DIR=/app/storage YOLO_CONFIG_DIR=/tmp/ultralytics OMP_NUM_THREADS=2
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/* && python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements-server.txt .
RUN pip install --no-cache-dir --no-deps torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
RUN pip install --no-cache-dir --index-url ${PIP_INDEX_URL} -r requirements-server.txt
COPY src ./src
ENV YOLO_AUTOINSTALL=false
RUN useradd --uid 10001 --create-home app && mkdir /app/storage && chown app:app /app/storage
USER app
CMD ["uvicorn","src.app:app","--host","0.0.0.0","--port","8000","--workers","1"]
