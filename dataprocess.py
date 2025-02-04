import shutil
import os
from tqdm import tqdm


import os
import shutil
from collections import defaultdict

import scipy.io
from tqdm import tqdm


ROOT = '../data/cd-fsl/cars'

SOURCE_DIR = os.path.join(ROOT, 'cars_train_by_folder')
TARGET_DIR = os.path.join(ROOT, 'cars_train2')

directory_list = os.listdir(SOURCE_DIR)

for dir in directory_list:
    if dir == './DS_Store':
        pass
    path = os.path.join(SOURCE_DIR, dir)
    for file in os.listdir(path):
        source = os.path.join(path, file)
        target = os.path.join(TARGET_DIR, file)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copy(source, target)