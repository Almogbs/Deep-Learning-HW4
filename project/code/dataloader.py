from torch.utils.data import Dataset
from torchvision import transforms
from pycocotools.coco import COCO
from PIL import Image
import torch
import os
import os.path as path
import csv 
import json 
import pandas as pd
import re

def parse_one_annot(path_to_data_file, filename):
    data = pd.read_csv(path_to_data_file)
    boxes_array = data[data["filename"] == filename][["xmin", "ymin", "xmax", "ymax"]].values
    return boxes_array

def getListOfFiles(dirName, data_folder):
    listOfFile = os.listdir(dirName)
    allFiles = list()
    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath, data_folder)
        else:
                fullPath = fullPath.replace(data_folder, '')
                allFiles.append(fullPath)  
    return allFiles    

def export_data(data, dtype, file):
    img = data[dtype] 
    data_file = open(file, 'w') 
    csv_writer = csv.writer(data_file) 
    count = 0
    for i in img: 
        if count == 0: 
            header = i.keys() 
            csv_writer.writerow(header) 
            count += 1
        csv_writer.writerow(i.values()) 
    data_file.close() 

def json_to_csv(json_path, json_dir):
    with open(json_path) as json_file: 
        data = json.load(json_file)

    export_data(data, 'images', os.path.join(json_dir, 'data_file.csv'))
    export_data(data, 'annotations', os.path.join(json_dir, 'data_file2.csv'))
    export_data(data, 'categories', os.path.join(json_dir, 'data_file3.csv'))

    data3 = pd.read_csv(os.path.join(json_dir,'data_file3.csv'))
    data2 = pd.read_csv(os.path.join(json_dir,'data_file2.csv'))
    data = pd.read_csv(os.path.join(json_dir,'data_file.csv'))
    data = data.rename(columns={"id": "image_id"})
    data3 = data3.rename(columns={"id": "category_id"})
    data2 = data2.merge(data, on='image_id', how='left')
    data2 = data2.merge(data3, on='category_id', how='left')
    data2 = data2.rename(columns={"supercategory": "class"})
    data2 = data2.rename(columns={"file_name": "filename"})
    data2['bbox'] = data2['bbox'].map(lambda x: x.lstrip('[').rstrip(']'))
    data2[['xmin','ymin','xmax','ymax']] = data2.bbox.str.split(",",expand=True,)
    #data2['xmin'] = data2['xmin'].astype(float).astype(int)
    #data2['xmax'] = data2['xmax'].astype(float).astype(int)
    #data2['ymin'] = data2['ymin'].astype(float).astype(int)
    #data2['ymax'] = data2['ymax'].astype(float).astype(int)
    #data2['xmax'] = data2['xmax'].astype(float) + data2['width'].astype(float)
    #data2['ymax'] = data2['ymax'].astype(float) + data2['height'].astype(float)
    data2['xmax'] = data2['xmin'].astype(float) + data2['xmax'].astype(float)
    data2['ymax'] = data2['ymin'].astype(float) + data2['ymax'].astype(float)

    data2 = data2[['filename','width','height','class','xmin','ymin','xmax','ymax']]
    data2.to_csv(os.path.join(json_dir,'annotations.csv'), index=False)

class TACODataset(Dataset):
    def __init__(self, root, data_file, device='cpu', transforms=None):
        self.root = root
        self.transforms = transforms
        self.device = device
        files = getListOfFiles(os.path.join(root), root)
        p = re.compile('batch')
        l2 = [ s for s in files if p.match(s) ]
        self.imgs = sorted(l2)
        self.path_to_data_file = data_file
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.imgs[idx])
        self.imgs[idx] = self.imgs[idx]
        img = Image.open(img_path).convert("RGB")
        box_list = parse_one_annot(self.path_to_data_file, self.imgs[idx])
        boxes = torch.as_tensor(box_list, dtype=torch.float32)
        num_objs = len(box_list)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:,0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes.to(self.device)
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
                 img, target = self.transforms(img, target)
        return img.to(self.device), target
    
    def __len__(self):
        return len(self.imgs)



