import argparse
import cv2
import os
import time
import sys
import numpy as np 

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.engine import DefaultPredictor
from centermask.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer

from demo.predictor import VisualizationDemo
from openimages_utils.data_dicts import get_train_dicts, get_val_dicts

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

    DatasetCatalog.register('openimages_val', get_val_dicts)
    MetadataCatalog.get('openimages_val').set(thing_classes=classes)
    openimages_val_metadata = MetadataCatalog.get('openimages_val')

    #Args
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', help='path to config file', required=True)
    parser.add_argument('--input', nargs='+', help='a list of space separated input images')
    parser.add_argument('--video', help='video to infer instances from')
    parser.add_argument('--output', help='a file or directory to save output visualizations.'
                        'If not given, will show output in an OpenCV window.',)
    parser.add_argument('--confidence_threshold', type=float, default=0.5, 
                        help='Minimum score for instance predictions to be shown.')
    parser.add_argument('--weights', help='.pth file to saved model weights')
    args = parser.parse_args()

    #Logs
    logger = setup_logger()
    logger.info("Arguments:" + str(args))
    setup_logger(name='fvcore')

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 300
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 300
    cfg.MODEL.RETINANET.NUM_CLASSES = 300
    cfg.MODEL.FCOS.NUM_CLASSES = 300
    cfg.DATASETS.TEST = ('openimages_val',)
    cfg.DATASETS.TRAIN = ('openimages_train',)

    predictor = DefaultPredictor(cfg)

    if args.input:
        for image_file in args.input:
            im = cv2.imread(image_file)
            start_time = time.time()
            outputs = predictor(im)
            # print(outputs['instances'].pred_classes)
            # print(outputs['instances'].pred_boxes)
            logger.info(
                "Detected {} instances in {:.2f}s".format(
                    len(outputs['instances'].pred_boxes), time.time()-start_time
                )
            )
            v = Visualizer(im[:, :, ::-1], metadata=openimages_val_metadata, scale=0.8)
            v = v.draw_instance_predictions(outputs['instances'].to('cpu'))

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(image_file))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                
                cv2.imwrite(out_filename, v.get_image()[:, :, ::-1])
                print("Instance saved to {}!".format(out_filename) )

            else:    
                cv2.imshow('instances', v.get_image()[:, :, ::-1])
                cv2.waitKey(0)
            
    elif args.video:
        assert args.input is None, "Cannot have both --input and --video!"

        video = cv2.VideoCapture(args.video)
        fps = video.get(cv2.CAP_PROP_FPS)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        writer = cv2.VideoWriter(args.output, fourcc, 20, (width, height), True)

        try:
            prop = cv2.CAP_PROP_FRAME_COUNT
            total = int(video.get(prop))
            print("[INFO] {} total frames in video".format(total))
        except:
            print("[INFO] could not determine no of frames in video")
            total = -1
        prog = 0
        while True:
            (grabbed, frame) = video.read()

            #no more frames present
            if not grabbed:
                break

            output = frame.copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            start_time = time.time()
            outputs = predictor(frame)
            inference_time = time.time() - start_time
            v = Visualizer(frame[:, :, ::-1], metadata=openimages_val_metadata, scale=0.8)
            v = v.draw_instance_predictions(outputs['instances'].to('cpu'))
            out_frame = v.get_image()[:, :, ::-1]
            out_frame = cv2.resize(out_frame, (width, height))
            #make contiguous
            out_frame = np.ascontiguousarray(out_frame)
            text = "Inference Time: {}".format(inference_time)
            cv2.putText(out_frame, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            out_frame = np.uint8(out_frame)
            if args.output:
                writer.write(out_frame)
                prog += 1
                sys.stdout.write("\r{:.2f}% COMPLETED".format((prog*100)/total))
      
            else:
                cv2.imshow('instance', out_frame)
                cv2.waitKey(1)
        writer.release()
        video.release()
    

            
        
            




        




