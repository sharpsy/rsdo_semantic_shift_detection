FROM python:3.10

COPY requirements.txt /app/
RUN pip install --no-cache-dir --disable-pip-version-check -r /app/requirements.txt
COPY data /data
COPY src /app/
