'''
dictionary that contains the following values:
file_name: full path of image file
sem_seg_file_name: full path to gt semantic segmentation
sem_seg: semantic segmentation in a 2D torch.tensor
height, width: image shape
image_id
annotations
    bbox: list of 4 numbers rep bounding box
    bbox_mode
    category_id
'''
import os
import csv
import cv2
import numpy as np
from datasets import pycocotools
from detectron2.data import DatasetCatalog, MetadataCatalog

#CHECK FIELDS FIRST
sem_seg_file_name = 'validation/challenge-2019-validation-segmentation-masks.csv'
img_dir = 'validation/images'

#returns a list of dictionaries, one dictionary per img
def convert_mask_png(image_file):
    image_folder = os.listdir("validation/" + str(image_file[0])
    image_path = os.path.join(image_folder,image_file)
    pycocotools.mask.encode(np.asarray(image_path, order="F"))

def get_dicts(img_dir):
    dataset_dicts = [] #list of dictionaries
    
    with open('challenge-2019-validation-segmentation-masks.csv', 'r') as csv_file:
        csv_reader_mask = csv.reader(csv_file, delimiter=',')
    with open('challenge-2019-validation-segmentation-bbox.csv', 'r') as csv_file:
        csv_reader_bbox = csv.reader(csv_file, delimiter=',')
    with open('challenge-2019-validation-segmentation-labels.csv', 'r') as csv_file:
        csv_reader_labels = csv.reader(csv_file, delimiter=',')
    
    for filename in os.listdir(img_dir):
        record = {}
        filename = os.path.join(img_dir,filename)
        height, width = cv2.imread(filename).shape[:2]
        
        #image id
        idx = os.path.splitext(filename)[0]
                
        record["file_name"] = filename
        record["image_id"] = idx
        record["height, width"] = height, width
        
        for row in csv_reader_bbox:
            if row[0] == idx:
                x_min = row[2]
                x_max = row[3]
                y_min = row[4]
                y_max = row[5]
        
        for row in csv_reader_labels:
            if row[0] == idx:
                cat_id = row[1]

        for row in csv_reader_mask:
            if row[1] == idx:
                mask_img_file = row[0]
 

        objs = {
            "bbox": [x_min, y_min, x_max, y_max],
            "bbox_mode": BoxMode.XYXY_ABS,
            "segmentation": dict,
            "category_id": cat_id,
            "iscrowd": 0
        }

        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

DatasetCatalog.register("my_dataset", get_dicts)
        
    