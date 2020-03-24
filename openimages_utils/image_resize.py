import json
import cv2
import os
from PIL import Image

#Only images found in masks are resized

folder_path = "C:\\Users\\Admin\\Documents\\detectron2\\datasets\\validation\\train"
mask_path = "C:\\Users\\Admin\\Documents\\detectron2\\datasets\\validation\\masks"

def resizing (image_path, mask_path):

    image = Image.open(image_path)
    mask = cv2.imread(mask_path, 0)
    height = mask.shape[0]
    width = mask.shape[1]
    resized = image.resize((width,height))
    return resized


mask_ids = []
for mask_file in os.listdir(mask_path):
    mask_id = os.path.splitext(mask_file)[0]
    mask_ids.append(mask_id)

i = 0

matching_img = []
matching_mask = []

while i < len(mask_ids):
    for image_file in os.listdir(folder_path):
        image_id = os.path.splitext(image_file)[0]
        if image_id in mask_ids[i]:
            matching_img.append(image_id)
            matching_mask.append(mask_ids[i])
            print(matching_img)
    i += 1


for j in range(len(matching_img)):
    resized = resizing(folder_path + '/' + matching_img[j] + '.jpg', mask_path + '/' + matching_mask[j] + '.png')
    resized = resized.convert("RGB")
    resized.save('train_resized/' + matching_img[j] + '.jpg', resized.format)
