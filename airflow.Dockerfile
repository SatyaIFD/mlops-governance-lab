FROM apache/airflow:3.4.0
RUN pip install --no-cache-dir pandas==2.3.3 pyarrow==26.0.1 httpx==0.28.1
