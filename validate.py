import argparse
import os
import random
from detectron2.engine import DefaultPredictor
from detectron2.data.datasets import load_coco_json
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, DatasetCatalog, MetadataCatalog
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--mode', help='"inference" or "evaluation')
args = parser.parse_args()

if __name__ == "__main__":
    
    cfg = get_cfg()

    def get_val_dicts():
        val_dicts = load_coco_json("validation.json", "C:\\Users\\Admin\\Documents\\detectron2\\datasets\\validation\\validation_resized", "open_images_val")
        return val_dicts

    DatasetCatalog.register("open_images_val", get_val_dicts)
    open_images_val_metadata = MetadataCatalog.get("open_images_val")

    #Visualizing datasets
    # val_dicts = get_val_dicts()
    # for d in random.sample(val_dicts, 10):
    #     img = cv2.imread(d["file_name"])
    #     print(d["file_name"])
    #     visualizer = Visualizer(img[:,:,::-1], metadata=open_images_val_metadata, scale=0.5)
    #     vis = visualizer.draw_dataset_dict(d)
    #     cv2.imshow("image", vis.get_image()[:,:,::-1])
    #     cv2.waitKey(0)
    #cfg.merge_from_file("configs\\COCO-InstanceSegmentation\\mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = "output\\model_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 601
    cfg.DATASETS.TEST = ("open_images_val",)

    predictor = DefaultPredictor(cfg)

    if args.mode == "inference":
        dataset_dicts = get_val_dicts()
        for d in random.sample(dataset_dicts, 10):
            im = cv2.imread(d["file_name"])
            outputs = predictor(im)
            print(outputs)
            v = Visualizer(im[:, :, ::-1],
                        metadata = open_images_val_metadata,
                        scale = 0.8,
                        )
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv2.imshow("image", v.get_image()[:, :, ::-1])
            cv2.waitKey(0)
    
    if args.mode == "evaluation":
        evaluator = COCOEvaluator("open_images_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
        val_loader = build_detection_test_loader(cfg, "open_images_val")
        inference_on_dataset(trainer.model, val_loader, evaluator)


