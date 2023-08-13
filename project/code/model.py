from ultralytics import YOLO

def get_model():
    model = YOLO('yolov8n.pt')

    return model


