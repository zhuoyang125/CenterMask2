import os
import numpy as np 
import json
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import random
from detectron2.utils.visualizer import Visualizer
import cv2
import matplotlib.pyplot as plt
#find exception
import linecache
import sys
import traceback

#Find exception
def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))


if __name__ == '__main__':
    try:
        register_coco_instances("openimage_dataset_train", {}, "train.json", "C:\\Users\\Admin\\Documents\\detectron2\\datasets\\validation\\train_resized")
        register_coco_instances("openimage_dataset_val", {}, "validation.json", "C:\\Users\\Admin\\Documents\\detectron2\\datasets\\validation\\validation_resized")

        openimage_dataset_train_metadata = MetadataCatalog.get("openimage_dataset_train")
        #openimage_dataset_val_metadata = MetadataCatalog.get("openimage_dataset_val")
        dataset_dicts_train = DatasetCatalog.get("openimage_dataset_train")
        #dataset_dicts_val = DatasetCatalog.get("openimage_dataset_val")

        # #Visualizing datasets
        # for d in random.sample(dataset_dicts_train, 20):
        #     img = cv2.imread(d["file_name"])
        #     print(d["file_name"])
        #     visualizer = Visualizer(img[:,:,::-1], metadata=openimage_dataset_train_metadata, scale=0.5)
        #     vis = visualizer.draw_dataset_dict(d)
        #     cv2.imshow("image", vis.get_image()[:,:,::-1])
        #     cv2.waitKey(0)

        #Training
        cfg = get_cfg()
        cfg.merge_from_file("configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.DATASETS.TRAIN = ("openimage_dataset_train",)
        cfg.DATASETS.TEST = ()
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.SOLVER.BASE_LR = 0.0025
        cfg.SOLVER.MAX_ITER = 120000
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 8
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 601

        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
    except KeyError:
        pass

        


