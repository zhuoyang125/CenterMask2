from data_dicts import get_train_dicts
import random
import numpy as np
from numpy import vectorize
import matplotlib.pyplot as plt

dataset_dicts = get_train_dicts()

data_distribution = np.zeros(300)

for d in dataset_dicts:
    for anno in d['annotations']:
        category = int(anno['category_id'])
        data_distribution[category] += 1


data_distribution = data_distribution / 316072
print(data_distribution)

plt.hist(data_distribution, range=(0, 0.06), bins=100)
plt.show()