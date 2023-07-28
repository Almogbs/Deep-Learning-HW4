from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pycocotools.coco import COCO
from PIL import Image
import torch
import os


def resize_image_and_annotations(image, annotations, target_size):
    # Resize the image
    image = image.resize(target_size, Image.BILINEAR)
    
    # Resize the bounding box coordinates in the annotations
    for annotation in annotations:
        bbox = annotation['bbox']
        x, y, w, h = bbox
        x_scale = target_size[0] / image.width
        y_scale = target_size[1] / image.height
        annotation['bbox'] = [x * x_scale, y * y_scale, w * x_scale, h * y_scale]
    
    return image, annotations


class TACODataset(Dataset):
    def __init__(self, coco_dataset, device='cpu', transform=None):
        self.coco = coco_dataset
        self.transform = transform

        # Obtain all image IDs
        self.image_ids = list(sorted(coco.getImgIds()))

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # Load the image and annotations for the given index
        image_info = self.coco.loadImgs(self.image_ids[idx])[0]
        image_path = os.path.join(data_root, image_info['file_name'])
        image = Image.open(image_path).convert('RGB')
        
        # Load the annotations
        annotation_ids = self.coco.getAnnIds(imgIds=self.image_ids[idx])
        annotations = self.coco.loadAnns(annotation_ids)
        
        # Apply transformations if provided
        if self.transform is not None:
            image, annotations = self.transform(image, annotations)
        
        image = image.to(device)
        annotations = [ann.to(device) for ann in annotations]

        return image, annotations


# Create the dataset
dataset = TACODataset(coco, transform=resize_image_and_annotations)

# Create the DataLoader
batch_size = 32
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Now you can iterate through the data_loader and access GPU-accelerated data
for images, annotations in data_loader:
    # Your training code here
    # The 'images' and 'annotations' variables are on the GPU
