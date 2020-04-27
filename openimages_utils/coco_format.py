import numpy as np
from imantics import Polygons, Mask
import cv2
from csv import reader
import os
import json

image_folder = "C:\\Users\\Admin\\Documents\\open-images-1\\train_resized"

coco_dict = {}
coco_dict['images'] = []
coco_dict['annotations'] = []
coco_dict['categories'] = []
with open('C:\\Users\\Admin\\Documents\\open-images-1\\challenge-2019-train-segmentation-masks.csv') as read_obj:
    csv_reader = reader(read_obj)
    csv_list = list(csv_reader)
with open('C:\\Users\\Admin\\Documents\\open-images-1\\challenge-2019-classes-description-segmentable.csv') as read_obj:
    csv_cat_reader = reader(read_obj)
    cat_list = list(csv_cat_reader)
        


for image_file in os.listdir(image_folder):
    image = cv2.imread(image_folder + '\\'+ image_file)
    height = image.shape[0]
    width = image.shape[1]
    image_id = os.path.splitext(image_file)[0]
    #print(image_id)
    for i in range(len(csv_list)):
        if csv_list[i][1] == image_id:
            mask_image_path = csv_list[i][0]
            label_name = csv_list[i][2]
            x_min_rel = float(csv_list[i][4])
            x_max_rel = float(csv_list[i][5])
            y_min_rel = float(csv_list[i][6])
            y_max_rel = float(csv_list[i][7])
            x_min = x_min_rel * width
            y_min = y_min_rel * height
            x_max = x_max_rel * width
            y_max = y_max_rel * height
            mask_image = cv2.imread('C:\\Users\\Admin\\Documents\\open-images-1\\masks' + '\\' + mask_image_path, 0)
            idx = os.path.splitext(mask_image_path)[0]
            print(idx)
            polygons = Mask(mask_image).polygons()
            coco_dict['annotations'].append({
            'id': idx,
            'image_id': image_id,
            'category_id': label_name,
            'segmentation': polygons.segmentation,
            #'bbox': [float(x_min), float(y_min), bbox_width, bbox_height],
            'bbox': [float(x_min),float(y_min),float(x_max),float(y_max)],
            'iscrowd': 0
            })
            coco_dict['images'].append({
                'file_name': str(image_file),
                'height': height,
                'width': width,
                'id': image_id
            })

for i in range(len(cat_list)):
        coco_dict['categories'].append({
            'id': cat_list[i][0],
            'name': str(cat_list[i][1]),
            'supercategory': str(cat_list[i][1])
        })
        

with open('train_1_XYXY.json', 'w') as fp:
    json.dump(coco_dict, fp)
