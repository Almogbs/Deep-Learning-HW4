import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import cv2
import os


def display_random_images_and_annotations(coco_obj, images_dir, num_images_to_display=6, rows=2, cols=3):
    image_ids = coco_obj.getImgIds()
    random.shuffle(image_ids)

    num_images = min(num_images_to_display, len(image_ids))
    fig, axs = plt.subplots(rows, cols, figsize=(12, 8))

    for i in range(rows):
        for j in range(cols):
            if i * cols + j < num_images:
                image_id = image_ids[i * cols + j]
                image_info = coco_obj.loadImgs(image_id)[0]
                image_path = os.path.join(images_dir, image_info['file_name'])

                image = cv2.imread(image_path)
                annotations = coco_obj.loadAnns(coco_obj.getAnnIds(imgIds=image_id))

                for ann in annotations:
                    bbox = ann['bbox']
                    category_id = ann['category_id']

                    category_info = coco_obj.loadCats(category_id)[0]
                    category_name = category_info['name']

                    x, y, w, h = bbox
                    cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 4)

                    label_text = f"{category_name}"
                    plt_text = axs[i, j].text(x, y - 5, label_text, fontsize=10, color='lime', backgroundcolor='none')

                axs[i, j].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), interpolation='nearest', aspect='auto')
                axs[i, j].axis('off')
                axs[i, j].set_title(f"Image ID: {image_id}")

    plt.tight_layout()
    plt.show()


def get_info_from_coco_obj(coco):
    cat_ids = coco.getCatIds()
    image_ids = coco.getImgIds()
    annotation_ids = coco.getAnnIds()

    num_cats = len(cat_ids)
    num_images = len(image_ids)
    num_annotations = len(annotation_ids)

    print("Number of categories in the dataset:", num_cats)
    print("Number of images in the dataset:", num_images)
    print("Number of annotations in the dataset:", num_annotations)


def get_category_histogram_from_coco_obj(coco):
    cat_ids = coco.getCatIds()
    annotation_ids = coco.getAnnIds()

    num_cats = len(cat_ids)
    cats = coco.loadCats(cat_ids)
    cat_names = [cat['name'] for cat in cats]

    cat_histogram = np.zeros(num_cats, dtype=int)
    for id in annotation_ids:
        info = coco.loadAnns(id)
        cat_histogram[info[0]['category_id']] += 1

    f, ax = plt.subplots(figsize=(5,15))

    df = pd.DataFrame({'Categories': cat_names, 'Number of annotations': cat_histogram})
    df = df.sort_values('Number of annotations', ascending=False)

    plot_1 = sns.barplot(x="Number of annotations", y="Categories", data=df,
                label="Total", color="b")

