import os
import torch
import numpy as np 
import json
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets import load_coco_json
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from centermask.config import get_cfg
from detectron2 import model_zoo
import random
from detectron2.utils.visualizer import Visualizer
import cv2
from openimages_utils.data_dicts import get_train_dicts
from detectron2.config import CfgNode as CN

classes = ['Toy', 'Dog', 'Flower', 'Boy', 'Skateboard', 'Hat', 'Man', 'Girl', 'Woman', 'Car', 'Human ear', 'Human mouth', 'Monkey', 'Jeans', 'Mug', 'Wheel', 'Juice', 'Person', 'Bicycle wheel', 'Squirrel', 'Zebra', 'Skyscraper', 'Bus', 'Guitar', 'Rose', 'Train', 'Cello', 'Suit', 'Tie', 'Computer keyboard', 'Sculpture', 'Drink', 'Balloon', 'Bronze sculpture', 'Dress', 'Van', 'Flowerpot', 'Trousers', 'Eagle', 'Cake', 'Sun hat', 'Book', 'Laptop', 'Bottle', 'Airplane', 'Skirt', 'Mouse', 'Fedora', 'Doughnut', 'Barrel', 'Vase', 'Falcon', 'Shirt', 'Cattle', 'Cocktail', 'Mobile phone', 'Bird', 'Bread', 'Beer', 'Motorcycle', 'Flag', 'Camera', 'Box', 'Cat', 'Swimwear', 'Couch', 'Goat', 'High heels', 'Carnivore', 'Horse', 'Parrot', 'Teddy bear', 'Handbag', 'Fish', 'Pillow', 'Swan', 'Duck', 'Goose', 'Sandwich', 'Wine', 'Scarf', 'Wok', 'Strawberry', 'Canoe', 'Mushroom', 'Shorts', 'Baseball glove', 'Ball', 'Chest of drawers', 'Coffee cup', 'Piano', 'Watch', 'Coin', 'Picture frame', 'Football', 'Banana', 'Truck', 'Billiard table', 'Plastic bag', 'Sunflower', 'Boot', 'Sea lion', 'Chicken', 'Ostrich', 'Teapot', 'Tap', 'Vehicle registration plate', 'Candle', 'Tank', 'Christmas tree', 'Coffee', 'Shark', 'Platter', 'Spoon', 'Sofa bed', 'Saxophone', 'Muffin', 'Lemon', 'Sombrero', 'Camel', 'Baseball bat', 'Rocket', 'Ambulance', 'Traffic sign', 'Taxi', 'Surfboard', 'Whale', 'Dolphin', 'Pen', 'Bowl', 'Goldfish', 'Hamburger', 'Pitcher', 'Jug', 'Luggage and bags', 'Tomato', 'Apple', 'Giraffe', 'Bagel', 'Elephant', 'Whiteboard', 'Barge', 'Saucer', 'Waste container', 'Tortoise', 'Owl', 'Sheep', 'Kite', 'Sock', 'Tablet computer', 'Traffic light', 'Zucchini', 'Suitcase', 'Bull', 'Studio couch', 'Carrot', 'Harbor seal', 'Cookie', 'Cowboy hat', 'Tea', 'Frog', 'Penguin', 'Miniskirt', 'Orange', 'Roller skates', 'Clock', 'Pig', 'Grapefruit', 'Oven', 'Pizza', 'Swim cap', 'Aircraft', 'Pumpkin', 'Broccoli', 'Tart', 'Loveseat', 'Potato', 'Mule', 'Pancake']

if __name__ == '__main__':
    #Register Datasets
    DatasetCatalog.register('openimages_train', get_train_dicts)
    MetadataCatalog.get('openimages_train').set(thing_classes=classes)
    openimages_train_metadata = MetadataCatalog.get('openimages_train')

    #Visualizing datasets
    # train_dicts = get_train_dicts()
    # for d in random.sample(train_dicts, 10):
    #     print(d)
    #     img = cv2.imread(d["file_name"])
    #     visualizer = Visualizer(img[:,:,::-1], metadata=openimages_train_metadata, scale=0.5)
    #     vis = visualizer.draw_dataset_dict(d)
    #     cv2.imshow("image", vis.get_image()[:,:,::-1])
    #     cv2.waitKey()

    cfg = get_cfg()

    #Training Configs
    cfg.merge_from_file('configs/centermask/Base-CenterMask-Lite-EfficientNet.yml')
    cfg.MODEL.WEIGHTS = 'C:\\Users\\Admin\\Documents\\detectron2\\projects\\CenterMask2\\output\\centermask\\CenterMask-Lite-Efficientnet-2x\\efficientnet-pretrained\\model_final_wo_solver_states.pth'
    cfg.DATASETS.TRAIN = ('openimages_train',)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.MAX_ITER = 300000
    cfg.SOLVER.BASE_LR = 0.00003
    cfg.SOLVER.GAMMA = 0.2
    cfg.SOLVER.STEPS = (150000, 220000,)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 8
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 179
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 179
    cfg.MODEL.RETINANET.NUM_CLASSES = 179
    cfg.MODEL.FCOS.NUM_CLASSES = 179

    #Sampler for imbalanced dataset
    cfg.DATALOADER.SAMPLER_TRAIN =  "RepeatFactorTrainingSampler"
    cfg.DATALOADER.REPEAT_THRESHOLD = 0.0005
    

    os.makedirs('./output/centermask/CenterMask-Lite-Efficientnet-2x', exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()