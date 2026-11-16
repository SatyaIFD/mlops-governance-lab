FROM python:3.13.14-slim
WORKDIR /app
RUN apt-get update && \
    apt-get upgrade -y && \
    rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir mlflow==3.18.0 msgpack==1.2.4 setuptools==84.2.1 urllib3==2.8.0 && \
    rm -rf /usr/local/lib/python3.13/site-packages/pip* /usr/local/lib/python3.13/ensurepip
RUN useradd -m -u 10001 mlopsuser && mkdir -p /app/mlruns /data && chown -R 10001:1000 /app /data && chmod -R 2775 /app /data
USER 10001
EXPOSE 5000
CMD ["mlflow", "server", "--backend-store-uri", "sqlite:////app/mlflow.db", "--default-artifact-root", "/app/mlruns", "--host", "0.0.0.0", "--port", "5000"]
