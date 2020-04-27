import json

with open('C:\\Users\\Admin\\Documents\\detectron2\\projects\\CenterMask2\\output\\openimages_val_coco_format.json') as f:
    annotations = json.load(f)

for anno in annotations["annotations"]:
    for i in (anno["segmentation"]):
        if len(i) < 6:
            print(i)