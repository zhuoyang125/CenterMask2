import csv
import json

with open('C:\\Users\\Admin\\Documents\\open-images-0\\challenge-2019-classes-description-segmentable.csv') as read_obj:
    csv_reader = csv.reader(read_obj)
    csv_list = list(csv_reader)

int_label = 0
int_label_dict = {}
for i in range(len(csv_list)):
    int_label_dict[csv_list[i][0]] = int_label
    int_label += 1
with open('int_category_labels.json', 'w') as f:
    json.dump(int_label_dict, f)
