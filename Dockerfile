FROM nvcr.io/nvidia/tritonserver:23.09-py3

RUN pip install opencv-python && \
    apt update && \
    apt install -y libgl1 && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update -y && \
    apt-get install build-essential cmake pkg-config

RUN pip install face_recognition==1.3.0

CMD ["tritonserver", "--model-repository=/models"]