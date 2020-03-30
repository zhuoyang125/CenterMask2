import json
from detectron2.structures import BoxMode

def get_train_dicts():
    #with open("/floyd/input/open-images/int_category_labels.json") as fp:
    with open('./datasets/int_category_labels.json') as fp:
        category_labels = json.load(fp)
    #with open("/floyd/input/open-images/train_XYXY.json") as f:
    with open('./datasets/train_XYXY.json') as f:
        img_anns = json.load(f)
    train_dicts = []

    for img in img_anns['images']:
        record = {}
        #record["file_name"] = "/floyd/input/open-images/train_resized/" + img["file_name"]
        record["file_name"] = "./datasets/train_resized/" + img["file_name"]
        record["height"] = img["height"]
        record["width"] = img["width"]
        record["image_id"] = img["id"]
        objs = []
        for annos in img_anns['annotations']:
            
            if annos["image_id"] == img["id"]:
                int_category_id = category_labels[annos["category_id"]]
                obj = {
                    "bbox": annos['bbox'],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": int_category_id,
                    "segmentation": annos['segmentation'],
                    "iscrowd": 0
                }
                objs.append(obj)
        record['annotations'] = objs
        train_dicts.append(record)
    return train_dicts

def get_val_dicts():
    with open("C:\\Users\\Admin\\Documents\\open-images\\int_category_labels.json") as fp:
        category_labels = json.load(fp)
    with open("C:\\Users\\Admin\\Documents\\open-images\\validation_XYXY.json") as f:
        img_anns = json.load(f)
    val_dicts = []

    for img in img_anns['images']:
        record = {}
        record["file_name"] = "C:\\Users\\Admin\\Documents\\open-images\\validation_resized\\" + img["file_name"]
        record["height"] = img["height"]
        record["width"] = img["width"]
        record["image_id"] = img["id"]
        objs = []
        for annos in img_anns['annotations']:
            
            if annos["image_id"] == img["id"]:
                int_category_id = category_labels[annos["category_id"]]
                obj = {
                    "bbox": annos['bbox'],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": int_category_id,
                    "segmentation": annos['segmentation'],
                    "iscrowd": 0
                }
                objs.append(obj)
        record['annotations'] = objs
        val_dicts.append(record)
    return val_dicts
