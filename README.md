# yolov8-triton-ensemble

### How to start

```console
docker build -t yolov8-triton .
docker run -d --gpus all -p 8000:8000 -v ~/yolov8-triton-ensemble/models:/models yolov8-triton
```
