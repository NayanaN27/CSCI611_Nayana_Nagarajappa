from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="coco128.yaml",
    epochs=30,
    imgsz=640,
    batch=16
)
