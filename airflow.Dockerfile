FROM apache/airflow:3.4.0@sha256:a8d9d19c563c768b2bc605ef26440f28a9164ea886064e99c599b057432ff572
RUN pip install --no-cache-dir pandas==2.3.3 pyarrow==26.0.1 httpx==0.28.1
