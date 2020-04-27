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

classes = ["Screwdriver",
"Light switch",
"Doughnut",
"Toilet paper",
"Wrench",
"Toaster",
"Tennis ball",
"Radish",
"Pomegranate",
"Kite",
"Table tennis racket",
"Hamster",
"Barge",
"Shower",
"Printer",
"Snowmobile",
"Fire hydrant",
"Limousine",
"Whale",
"Microwave oven",
"Asparagus",
"Lion",
"Spatula",
"Torch",
"Volleyball",
"Ambulance",
"Chopsticks",
"Raccoon",
"Blue jay",
"Lynx",
"Dice",
"Filing cabinet",
"Ruler",
"Power plugs and sockets",
"Bell pepper",
"Binoculars",
"Pretzel",
"Hot dog",
"Missile",
"Common fig",
"Croissant",
"Adhesive tape",
"Slow cooker",
"Dog bed",
"Harpsichord",
"Billiard table",
"Alpaca",
"Harbor seal",
"Grape",
"Nail",
"Paper towel",
"Alarm clock",
"Guacamole",
"Starfish",
"Zebra",
"Segway",
"Sea turtle",
"Scissors",
"Rhinoceros",
"Kangaroo",
"Jaguar",
"Leopard",
"Dumbbell",
"Envelope",
"Winter melon",
"Teapot",
"Camel",
"Beaker",
"Brown bear",
"Toilet",
"Teddy bear",
"Briefcase",
"Stop sign",
"Tiger",
"Cabbage",
"Giraffe",
"Polar bear",
"Shark",
"Rabbit",
"Swim cap",
"Pressure cooker",
"Kitchen knife",
"Submarine sandwich",
"Flashlight",
"Penguin",
"Snake",
"Zucchini",
"Bat",
"Food processor",
"Ostrich",
"Sea lion",
"Goldfish",
"Elephant",
"Rocket",
"Mouse",
"Oyster",
"Digital clock",
"Otter",
"Dolphin",
"Punching bag",
"Corded phone",
"Tennis racket",
"Pancake",
"Mango",
"Crocodile",
"Waffle",
"Computer mouse",
"Kettle",
"Tart",
"Oven",
"Banana",
"Cheetah",
"Raven",
"Frying pan",
"Pear",
"Fox",
"Skateboard",
"Rugby ball",
"Watermelon",
"Flute",
"Canary",
"Door handle",
"Saxophone",
"Burrito",
"Suitcase",
"Roller skates",
"Dagger",
"Seat belt",
"Washing machine",
"Jet ski",
"Sombrero",
"Pig",
"Drinking straw",
"Peach",
"Tortoise",
"Towel",
"Tablet computer",
"Cucumber",
"Mule",
"Potato",
"Frog",
"Bear",
"Lighthouse",
"Belt",
"Baseball bat",
"Racket",
"Sword",
"Bagel",
"Goat",
"Lizard",
"Parrot",
"Owl",
"Turkey",
"Cello",
"Knife",
"Handgun",
"Carrot",
"Hamburger",
"Grapefruit",
"Tap",
"Tea",
"Bull",
"Turtle",
"Bust",
"Monkey",
"Wok",
"Broccoli",
"Pitcher",
"Whiteboard",
"Squirrel",
"Jug",
"Woodpecker",
"Pizza",
"Surfboard",
"Sofa bed",
"Sheep",
"Candle",
"Muffin",
"Cookie",
"Apple",
"Chest of drawers",
"Skull",
"Chicken",
"Loveseat",
"Baseball glove",
"Piano",
"Waste container",
"Barrel",
"Swan",
"Taxi",
"Lemon",
"Pumpkin",
"Sparrow",
"Orange",
"Tank",
"Sandwich",
"Coffee",
"Juice",
"Coin",
"Pen",
"Watch",
"Eagle",
"Goose",
"Falcon",
"Christmas tree",
"Sunflower",
"Vase",
"Football",
"Canoe",
"High heels",
"Spoon",
"Mug",
"Swimwear",
"Duck",
"Cat",
"Tomato",
"Cocktail",
"Clock",
"Cowboy hat",
"Miniskirt",
"Cattle",
"Strawberry",
"Bronze sculpture",
"Pillow",
"Squash",
"Traffic light",
"Saucer",
"Reptile",
"Cake",
"Plastic bag",
"Studio couch",
"Beer",
"Scarf",
"Coffee cup",
"Wine",
"Mushroom",
"Traffic sign",
"Camera",
"Rose",
"Couch",
"Handbag",
"Fedora",
"Sock",
"Computer keyboard",
"Mobile phone",
"Ball",
"Balloon",
"Horse",
"Boot",
"Fish",
"Backpack",
"Skirt",
"Van",
"Bread",
"Glove",
"Dog",
"Airplane",
"Motorcycle",
"Drink",
"Book",
"Train",
"Flower",
"Carnivore",
"Human ear",
"Toy",
"Box",
"Truck",
"Wheel",
"Aircraft",
"Bus",
"Human mouth",
"Sculpture",
"Shirt",
"Hat",
"Vehicle registration plate",
"Guitar",
"Sun hat",
"Bottle",
"Luggage and bags",
"Trousers",
"Bicycle wheel",
"Suit",
"Bowl",
"Man",
"Flowerpot",
"Laptop",
"Boy",
"Picture frame",
"Bird",
"Car",
"Shorts",
"Woman",
"Platter",
"Tie",
"Girl",
"Skyscraper",
"Person",
"Flag",
"Jeans",
"Dress"]

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
    cfg.merge_from_file("configs/centermask/centermask_lite_V_19_eSE_FPN_ms_4x.yaml")
    cfg.DATASETS.TRAIN = ("openimages_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.MODEL.WEIGHTS = 'C:\\Users\\Admin\\Documents\\detectron2\\projects\\CenterMask2\\output\\centermask\\CenterMask-Lite-V-19-ms-4x\\model_final_wo_solver_states.pth'
    cfg.SOLVER.BASE_LR = 0.0001
    cfg.SOLVER.GAMMA = 0.2
    cfg.SOLVER.STEPS = (100000,180000,)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 16
    cfg.SOLVER.MAX_ITER = 200000
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 300
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 300
    cfg.MODEL.RETINANET.NUM_CLASSES = 300
    cfg.MODEL.FCOS.NUM_CLASSES = 300


    #Gradient Clipping
    # cfg.SOLVER.CLIP_GRADIENTS = CN({"ENABLED": True})
    # cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
    # cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 0.50

    #Sampler for imbalanced dataset
    cfg.DATALOADER.SAMPLER_TRAIN =  "RepeatFactorTrainingSampler"
    cfg.DATALOADER.REPEAT_THRESHOLD = 0.0005
    
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    
