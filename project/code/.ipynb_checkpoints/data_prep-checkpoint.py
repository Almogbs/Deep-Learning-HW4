import yaml

def map_to_new_categories(cat_id, cat_names, categories_inv, categories_idx):
    cat_name = cat_names[cat_id]
    if cat_name in categories_inv.keys():
        cat = categories_inv[cat_name]
        cat_id = categories_idx[cat]
        return cat_id


def coco_to_yolo_fmt(ann):
    curr_bbox = ann['bbox']
    x = curr_bbox[0]
    y = curr_bbox[1]
    w = curr_bbox[2]
    h = curr_bbox[3]

    x_mid = (x + (x + w)) / 2
    y_mid = (y + (y + h)) / 2

    return x_mid, y_mid, w, h


def convert_taco_annotations_to_yaml(yolo_dataset_path, output_yaml, new_categories_map):
    class_names = list(new_categories_map.keys())

    yaml_dict = {
        'train': yolo_dataset_path + "/train/images",
        'val': yolo_dataset_path + "/val/images",
        'test': yolo_dataset_path + "/test/images",
        'names': class_names
    }

    with open(output_yaml, 'w') as yaml_file:
        yaml.dump(yaml_dict, yaml_file)
