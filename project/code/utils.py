from torchvision import transforms
import matplotlib.patches as patches
from matplotlib.patches import Polygon, Rectangle
import matplotlib.pyplot as plt
from PIL import Image, ExifTags
from matplotlib.collections import PatchCollection
import colorsys


def display_image(dataset, idx):
    image, annotations = dataset[idx]
    image = Image.fromarray(image.mul(255).permute(1, 2,0).byte().cpu().numpy())

    fig,ax = plt.subplots(1, figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image)

    # Plot annotations on the image as polygons
    for box in annotations['boxes'].cpu():
            [x, y, w, h] = box
            rect = Rectangle((x,y),w-x,h-y,linewidth=2,
                             facecolor='none', alpha=0.7, linestyle = '--')
            ax.add_patch(rect)

    plt.show()
    #0547666611