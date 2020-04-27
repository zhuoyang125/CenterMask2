import argparse
import os
import random
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.data.datasets import load_coco_json
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader, DatasetCatalog, MetadataCatalog
import cv2
from openimages_utils.data_dicts import get_val_dicts, get_train_dicts

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

if __name__ == "__main__":
    
    #Add arguments
    parser = argparse.ArgumentParser(description='choose args to infer instances frm sample test images OR evaluate model performance on test dataset')
    parser.add_argument('--configs', help='path to model config file', required=True)
    parser.add_argument('--model_pth', help='path to .pth file', required=True)
    parser.add_argument('--mode', help='choose "infer" or "evaluate" ')
    parser.add_argument('--threshold', type=float, help='confidence threshold level, float bet 0 and 1', default=0.5)
    args = parser.parse_args()
    
    #Get Datasets
    DatasetCatalog.register('openimages_val', get_val_dicts)
    MetadataCatalog.get('openimages_val').set(thing_classes=classes)
    openimages_val_metadata = MetadataCatalog.get('openimages_val')

    DatasetCatalog.register('openimages_train', get_train_dicts)
    MetadataCatalog.get('openimages_train').set(thing_classes=classes)
    openimages_train_metadata = MetadataCatalog.get('openimages_train')

    #Visualizing datasets
    # val_dicts = get_val_dicts()
    # for d in random.sample(val_dicts, 10):
    #     img = cv2.imread(d["file_name"])
    #     print(d["file_name"])
    #     visualizer = Visualizer(img[:,:,::-1], metadata=openimages_val_metadata, scale=0.5)
    #     vis = visualizer.draw_dataset_dict(d)
    #     cv2.imshow("image", vis.get_image()[:,:,::-1])
    #     cv2.waitKey(0)

    cfg = get_cfg()
    cfg.merge_from_file(str(args.configs))
    cfg.MODEL.WEIGHTS = str(args.model_pth)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.threshold
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 300
    cfg.DATASETS.TEST = ("openimages_val",)
    cfg.DATASETS.TRAIN = ("openimages_train",)

    #infer image instances
    predictor = DefaultPredictor(cfg)

    # Visualizing Datasets
    # val_dicts = get_val_dicts()
    # for d in random.sample(val_dicts, 10):
    #     im = cv2.imread(d['file_name'])
    #     outputs = predictor(im)
    #     v = Visualizer(im[:, :, ::-1],
    #                     metadata = openimages_val_metadata,
    #                     scale = 0.8,
    #                     instance_mode = ColorMode.IMAGE_BW)
    #     v = v.draw_instance_predictions(outputs['instances'].to('cpu'))
    #     cv2.imshow('image', v.get_image()[:, :, ::-1])
    #     cv2.waitKey(0)
    
    if args.mode == 'infer':
        val_dicts = get_val_dicts()
        for d in random.sample(val_dicts, 10):
            im = cv2.imread(d['file_name'])
            outputs = predictor(im)
            v = Visualizer(im[:, :, ::-1],
                            metadata = openimages_val_metadata,
                            scale = 0.8,
                            instance_mode = ColorMode.IMAGE_BW)
            v = v.draw_instance_predictions(outputs['instances'].to('cpu'))
            cv2.imshow('image', v.get_image()[:, :, ::-1])
            cv2.waitKey(0)
    
    if args.mode == 'evaluate':
        print(cfg.DATASETS.TEST[0])
        trainer = DefaultTrainer(cfg)
        model = trainer.build_model(cfg)
        DetectionCheckpointer(model).load(args.model_pth)
        evaluator = COCOEvaluator(cfg.DATASETS.TEST[0], cfg, False, output_dir="./output/")
        val_loader = build_detection_test_loader(cfg, "openimages_val")
        inference_on_dataset(model, val_loader, evaluator)
        






