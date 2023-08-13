import yaml

def map_to_new_categories(cat_id: int, cat_names: dict, categories_inverted: dict, categories_idx: dict) -> int:
    """
    @param cat_id: TACO category ID
    @param cat_names: category names
    @param categories_inverted: new category inverted mapping
    @param categories_idx: new category idx mapping
    @return: the new category ID
    """
    cat_name = cat_names[cat_id]
    if cat_name in categories_inverted.keys():
        cat = categories_inverted[cat_name]
        cat_id = categories_idx[cat]
        return cat_id


def coco_to_yolo_fmt(ann: dict) -> (float, float, float, float):
    """
    @param ann: COCO annotations
    @return: tuple of middle point coor, weight and height
    """
    curr_bbox = ann['bbox']
    x = curr_bbox[0]
    y = curr_bbox[1]
    w = curr_bbox[2]
    h = curr_bbox[3]

    x_mid = (x + (x + w)) / 2
    y_mid = (y + (y + h)) / 2

    return x_mid, y_mid, w, h


def convert_taco_annotations_to_yaml(yolo_dataset_path: str, output_yaml: str, new_categories_map: dict) -> None:
    """
    @param yolo_dataset_path:path to the dataset
    @param output_yaml: yaml file output name
    @param new_categories_map: new category mapping
    """
    class_names = list(new_categories_map.keys())

    yaml_dict = {
        'train': yolo_dataset_path + "/train/images",
        'val': yolo_dataset_path + "/val/images",
        'test': yolo_dataset_path + "/test/images",
        'names': class_names
    }

    with open(output_yaml, 'w') as yaml_file:
        yaml.dump(yaml_dict, yaml_file)
