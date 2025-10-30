from ultralytics import YOLO
import torch
import numpy as np

# import pdb; pdb.set_trace()
model = YOLO("mtq_int8_layer.engine")  

metrics = model.val(data="/workspace/yolov11/coco/coco.yaml", imgsz=640, batch=1, conf=0.25, iou=0.6, device="0",half=True)
