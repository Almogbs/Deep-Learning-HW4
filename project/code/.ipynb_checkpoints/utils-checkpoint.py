import os.path
import json
import argparse
import numpy as np
import random
import datetime as dt
import copy


def split_dataset(ann_path, test_ratio, valid_ratio):
with open(ann_input_path, 'r') as f:
    dataset = json.loads(f.read())

anns = dataset['annotations']
scene_anns = dataset['scene_annotations']
imgs = dataset['images']
nr_images = len(imgs)

nr_testing_images = int(nr_images*args.test_percentage*0.01+0.5)
nr_nontraining_images = int(nr_images*(args.test_percentage+args.val_percentage)*0.01+0.5)


for i in range(args.nr_trials):
    random.shuffle(imgs)

    # Add new datasets
    train_set = {
        'info': None,
        'images': [],
        'annotations': [],
        'scene_annotations': [],
        'licenses': [],
        'categories': [],
        'scene_categories': [],
    }
    train_set['info'] =  dataset['info']
    train_set['categories'] = dataset['categories']
    train_set['scene_categories'] = dataset['scene_categories']

    val_set = copy.deepcopy(train_set)
    test_set = copy.deepcopy(train_set)

    test_set['images'] = imgs[0:nr_testing_images]
    val_set['images'] = imgs[nr_testing_images:nr_nontraining_images]
    train_set['images'] = imgs[nr_nontraining_images:nr_images]

    # Aux Image Ids to split annotations
    test_img_ids, val_img_ids, train_img_ids = [],[],[]
    for img in test_set['images']:
        test_img_ids.append(img['id'])

    for img in val_set['images']:
        val_img_ids.append(img['id'])

    for img in train_set['images']:
        train_img_ids.append(img['id'])

    # Split instance annotations
    for ann in anns:
        if ann['image_id'] in test_img_ids:
            test_set['annotations'].append(ann)
        elif ann['image_id'] in val_img_ids:
            val_set['annotations'].append(ann)
        elif ann['image_id'] in train_img_ids:
            train_set['annotations'].append(ann)

    # Split scene tags
    for ann in scene_anns:
        if ann['image_id'] in test_img_ids:
            test_set['scene_annotations'].append(ann)
        elif ann['image_id'] in val_img_ids:
            val_set['scene_annotations'].append(ann)
        elif ann['image_id'] in train_img_ids:
            train_set['scene_annotations'].append(ann)

    # Write dataset splits
    ann_train_out_path = args.dataset_dir + '/' + 'annotations_' + str(i) +'_train.json'
    ann_val_out_path   = args.dataset_dir + '/' + 'annotations_' + str(i) + '_val.json'
    ann_test_out_path  = args.dataset_dir + '/' + 'annotations_' + str(i) + '_test.json'

    with open(ann_train_out_path, 'w+') as f:
        f.write(json.dumps(train_set))

    with open(ann_val_out_path, 'w+') as f:
        f.write(json.dumps(val_set))

    with open(ann_test_out_path, 'w+') as f:
        f.write(json.dumps(test_set))
        

def resize_image(image, min_dim=None, max_dim=None, min_scale=None, mode="square"):
    """Resizes an image keeping the aspect ratio unchanged.

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    min_scale: if provided, ensure that the image is scaled up by at least
        this percent even if min_dim doesn't require it.
    mode: Resizing mode.
        none: No resizing. Return the image unchanged.
        square: Resize and pad with zeros to get a square image
            of size [max_dim, max_dim].
        pad64: Pads width and height with zeros to make them multiples of 64.
               If min_dim or min_scale are provided, it scales the image up
               before padding. max_dim is ignored in this mode.
               The multiple of 64 is needed to ensure smooth scaling of feature
               maps up and down the 6 levels of the FPN pyramid (2**6=64).
        crop: Picks random crops from the image. First, scales the image based
              on min_dim and min_scale, then picks a random crop of
              size min_dim x min_dim. Can be used in training only.
              max_dim is not used in this mode.

    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Keep track of image dtype and return results in the same dtype
    image_dtype = image.dtype
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]
    crop = None

    if mode == "none":
        return image, window, scale, padding, crop

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    if min_scale and scale < min_scale:
        scale = min_scale

    # Does it exceed max dim?
    if max_dim and mode == "square":
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max

    # Resize image using bilinear interpolation
    if scale != 1:
        image = skimage.transform.resize(
            image, (round(h * scale), round(w * scale)),
            order=1, mode="constant", preserve_range=True)

    # Need padding or cropping?
    if mode == "square":
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "pad64":
        h, w = image.shape[:2]
        # Both sides must be divisible by 64
        assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
        # Height
        if h % 64 > 0:
            max_h = h - (h % 64) + 64
            top_pad = (max_h - h) // 2
            bottom_pad = max_h - h - top_pad
        else:
            top_pad = bottom_pad = 0
        # Width
        if w % 64 > 0:
            max_w = w - (w % 64) + 64
            left_pad = (max_w - w) // 2
            right_pad = max_w - w - left_pad
        else:
            left_pad = right_pad = 0
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "crop":
        # Pick a random crop
        h, w = image.shape[:2]
        y = random.randint(0, (h - min_dim))
        x = random.randint(0, (w - min_dim))
        crop = (y, x, min_dim, min_dim)
        image = image[y:y + min_dim, x:x + min_dim]
        window = (0, 0, min_dim, min_dim)
    else:
        raise Exception("Mode {} not supported".format(mode))
    return image.astype(image_dtype), window, scale, padding, crop


def resize_mask(mask, scale, padding, crop=None):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the mask, are resized consistently.

    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (0, 0)]
    """
    # Suppress warning from scipy 0.13.0, the output shape of zoom() is
    # calculated with round() instead of int()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
    if crop is not None:
        y, x, h, w = crop
        mask = mask[y:y + h, x:x + w]
    else:
        mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask
