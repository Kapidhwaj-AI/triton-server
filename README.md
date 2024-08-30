# yolov8-triton-ensemble

### How to start

```console
docker build -t triton .
docker run -d --gpus all -p8000:8000 -p8001:8001 -p8002:8002 -v ~/triton-server/models:/models triton
```
