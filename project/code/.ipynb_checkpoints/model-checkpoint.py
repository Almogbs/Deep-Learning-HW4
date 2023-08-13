from ultralytics import YOLO
import json


def get_model():
    """
    @return: yolov8 namo model
    """
    model = YOLO('yolov8n.pt')

    return model


def get_trained_model(dir: str, name: str, output_yaml: str, params: dict = {}) -> YOLO:
    """
    @param dir: model dir name
    @param name: model name
    @param params: extra params for the training
    @return: yolo trained model
    """
    try:
        model = YOLO(f"./runs/detect/{dir}/weights/best.pt")
        print(f"Cached {name} model was found!")
        return model
    except FileNotFoundError as e:
        print(f"No cached {name} model was found! Training:")
        model = get_model()
        # Train the model
        results = model.train(data=output_yaml, epochs=100, imgsz=640, cache=True, name=dir, **params)
        return model


def get_model_res_dict(dir: str, name: str, model: YOLO) -> dict:
    """
    @param dir: model dir name
    @param name: model name
    @param model: yolo trained model
    @return: res_dict with mAP50 and mAP50-95
    """
    res_dict = {}

    try:
        res_dict = json.load(open(f'./runs/detect/{dir}/test_res.json', 'r'))
        print(f"Cached {name} model rest results found!")
    except FileNotFoundError as e:
        print(f"Cached {name} model rest results not found! Evaluating:")
        res = model.val(split='test') 
        res_dict["mAP50-95"] = res.box.map
        res_dict["mAP50"] = res.box.map50
        json.dump(res_dict, open(f'./runs/detect/{dir}/test_res.json', 'w'))

    return res_dict
