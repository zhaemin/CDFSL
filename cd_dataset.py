"""
source : https://github.com/sungnyun/understanding-cdfsl
        https://github.com/alinlab/PsCo
All dataset classes in unified `torchvision.datasets.ImageFolder` format!
"""

import os
import sys
from PIL import Image

import numpy as np
import pandas as pd
import torch
import json
from torchvision.datasets import ImageFolder

#from configs import *


class CropDiseaseDataset(ImageFolder):
    name = "CropDisease"

    def __init__(self, root='../data/cd-fsl/CropDisease/dataset/train/', *args, **kwargs):
        super().__init__(root, *args, **kwargs)


class EuroSATDataset(ImageFolder):
    name = "EuroSAT"

    def __init__(self, root='../data/cd-fsl/EuroSAT/', *args, **kwargs):
        super().__init__(root, *args, **kwargs)


class ISICDataset(ImageFolder):
    name = "ISIC"
    """
    Implementation note: functions for finding data files have been customized so that data is selected based on
    the given CSV file.
    """

    def __init__(self, root='../data/cd-fsl/ISIC/', *args, **kwargs):
        csv_path = os.path.join(root, "ISIC2018_Task3_Training_GroundTruth.csv")
        self.metadata = pd.read_csv(csv_path)
        super().__init__(root, *args, **kwargs)

    def make_dataset(self, root, *args, **kwargs):
        paths = np.asarray(self.metadata.iloc[:, 0]) # ex -> ISIC_0024306
        labels = np.asarray(self.metadata.iloc[:, 1:]) # ex -> 0.0,1.0,0.0,0.0,0.0,0.0,0.0
        labels = (labels != 0).argmax(axis=1)

        samples = []
        for path, label in zip(paths, labels):
            path = os.path.join(root, path + ".jpg")
            samples.append((path, label))
        samples.sort()

        return samples

    def find_classes(self, _):
        classes = self.metadata.columns[1:].tolist() # MEL,NV,BCC,AKIEC,BKL,DF,VASC
        classes.sort()
        class_to_idx = dict()
        for i, cls in enumerate(classes):
            class_to_idx[cls] = i
        return classes, class_to_idx

    _find_classes = find_classes  # compatibility with earlier versions


class ChestXDataset(ImageFolder):
    name = "ChestX"
    """
    Implementation note: functions for finding data files have been customized so that data is selected based on
    the given CSV file.
    """

    def __init__(self, root='../data/cd-fsl/ChestX/', *args, **kwargs):
        csv_path = os.path.join(root, "Data_Entry_2017.csv")
        images_root = os.path.join(root, "images")
        # self.used_labels = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
        #                     "Pneumothorax"]
        self.used_labels = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
                            "Pneumothorax"]
        self.labels_maps = {"Atelectasis": 0, "Cardiomegaly": 1, "Effusion": 2, "Infiltration": 3, "Mass": 4,
                            "Nodule": 5, "Pneumothorax": 6}
        self.metadata = pd.read_csv(csv_path)
        super().__init__(images_root, *args, **kwargs)

    def make_dataset(self, root, *args, **kwargs):
        samples = []
        paths = np.asarray(self.metadata.iloc[:, 0]) # ex -> 00000001_000.png
        labels = np.asarray(self.metadata.iloc[:, 1]) # ex -> Cardiomegaly
        for path, label in zip(paths, labels):
            label = label.split("|")
            if len(label) == 1 and label[0] != "No Finding" and label[0] != "Pneumonia" and label[
                0] in self.used_labels:
                path = os.path.join(root, path)
                label = self.labels_maps[label[0]]
                samples.append((path, label))
        samples.sort()
        return samples

    def find_classes(self, _):
        return self.used_labels, self.labels_maps

    _find_classes = find_classes  # compatibility with earlier versions


class JSONImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, root, split, transform, depth=0):
        super().__init__()

        with open(f'../data/cd-fsl/splits/{dataset}/{split}.json', 'r') as f:
            dir_list = json.load(f)
        class_to_idx = {category: i for i, category in enumerate(dir_list['label_names'])}

        self.samples = []
        self.targets = dir_list['image_labels']
        _, self.targets = torch.tensor(self.targets).unique(return_inverse=True)
        
        if dataset == 'cars':
            for file_name, index in zip(dir_list['image_names'], dir_list['image_labels']):
                #new_file_name = file_name.split('/')[0] + '/0' + file_name.split('/')[1]
                #file_name = new_file_name
                file_path = os.path.join(root, file_name)
                self.samples.append((file_path, int(index)))
        else:
            for path in dir_list['image_names']:
                if dataset == 'cub':
                    new_path = path.split('/')[1] + '/' + path.split('/')[2]
                    path = new_path
                file_path = os.path.join(root, path)
                category = path.split('/')[depth]
                self.samples.append((file_path, class_to_idx[category]))

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        filename, y = self.samples[index]
        img = Image.open(filename, mode='r').convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, y


class CarsDataset(JSONImageDataset):
    name = "cars"

    def __init__(self, root='../data/cd-fsl/cars', split="novel", transform=None, depth=1):
        super().__init__("cars", root, split, transform, depth)


class CUBDataset(JSONImageDataset):
    name = "cub"

    def __init__(self, root='../data/cd-fsl/cub/images', split="novel", transform=None, depth=0):
        super().__init__("cub", root, split, transform, depth)


class PlacesDataset(JSONImageDataset):
    name = "places"

    def __init__(self, root='../data/cd-fsl/places', split="novel", transform=None, depth=1):
        super().__init__("places", root, split, transform, depth)


class PlantaeDataset(JSONImageDataset):
    name = "plantae"

    def __init__(self, root='../data/cd-fsl/plantae', split="novel", transform=None, depth=0):
        super().__init__("plantae", root, split, transform, depth)


def load_crossdomain_dataset(dataset_name, transform):
    dataset_classes = [
        CropDiseaseDataset,
        EuroSATDataset,
        ISICDataset,
        ChestXDataset,
        CarsDataset,
        CUBDataset,
        PlacesDataset,
        PlantaeDataset
    ]

    dataset_class_map = {
        cls.name: cls for cls in dataset_classes
    }
    
    dataset_cls = dataset_class_map[dataset_name]
    
    return dataset_cls(transform = transform)