import torch
from ultralytics import YOLO

def train():
    model = YOLO("yolov8n.pt")  # or path to pretrained
    model.train(
        data="synthetic_data.yaml",
        imgsz=512,
        epochs=50,
        batch=16,
        project="yolo_runs",
        name="synthetic_defect"
    )

if __name__ == "__main__":
    train()
