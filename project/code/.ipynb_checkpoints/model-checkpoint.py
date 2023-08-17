from ultralytics import YOLO
import shutil
import json
import os


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
    #hacky but necessary in order to be able to not save the weights...
    if os.path.exists(os.getcwd() + f"/project/results/{dir}/results.png"):
        return None
    
    try:
        model = YOLO(os.getcwd() + f"/project/results/{dir}/weights/best.pt")
        print(f"Cached {name} model was found!")
        return model
    except FileNotFoundError as e:
        print(f"No cached {name} model was found! Training:")
        model = get_model()
        # Train the model
        results = model.train(data=output_yaml, imgsz=640, cache=True, name=dir, **params)
        try:
            os.makedirs(os.getcwd() + f"/project/results/{dir}/weights")
        except FileExistsError as e:
            pass

        try:
            shutil.copyfile(os.getcwd() + f"/runs/detect/{dir}/results.png", os.getcwd() + f"/project/results/{dir}/results.png")
            shutil.copyfile(os.getcwd() + f"/runs/detect/{dir}/weights/best.pt", os.getcwd() + f"/project/results/{dir}/weights/best.pt")
        except FileNotFoundError as e:
            shutil.copyfile(os.getcwd() + f"/runs/detect/{dir}2/results.png", os.getcwd() + f"/project/results/{dir}/results.png")
            shutil.copyfile(os.getcwd() + f"/runs/detect/{dir}2/weights/best.pt", os.getcwd() + f"/project/results/{dir}/weights/best.pt")
        return model


def get_model_res_dict(dir: str, name: str, model: YOLO, set: str="val") -> dict:
    """
    @param dir: model dir name
    @param name: model name
    @param model: yolo trained model
    @return: res_dict with mAP50 and mAP50-95 on set
    """
    res_dict = {}

    try:
        res_dict = json.load(open(os.getcwd() + f'/project/results/{dir}/{set}_res.json', 'r'))
        print(f"Cached {name} model {set} results found!")
    except FileNotFoundError as e:
        print(f"Cached {name} model {set} results not found! Evaluating:")
        res = model.val(split=set) 
        res_dict["mAP50-95"] = res.box.map
        res_dict["mAP50"] = res.box.map50
        json.dump(res_dict, open(os.getcwd() + f'/project/results/{dir}/{set}_res.json', 'w'))

    return res_dict
