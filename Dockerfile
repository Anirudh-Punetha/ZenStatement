FROM nvidia/cuda:12.6.2-devel-ubuntu22.04

RUN apt-get update \
&& DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    git \
    curl \
    python3-pip \
&& pip3 install -U --no-cache-dir pip setuptools wheel \
&& rm -rf /var/lib/apt/lists/*

WORKDIR /opt/code

COPY requirements.txt api.py .

RUN pip3 install -U --no-cache-dir -r ani_test_req.txt && rm -rf /root/.cache

EXPOSE 80
EXPOSE 8000

CMD uvicorn api:app --host 0.0.0.0 --port 8000 --reload --workers 3 --timeout-keep-alive 60 --log-level info