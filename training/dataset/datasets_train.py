import csv
import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image


class ImageDataset_Train(Dataset):
    '''
    Data format in .csv file each line:
    /path/to/image.jpg,label,uni_label
    '''

    def __init__(self, csv_file, owntransforms, state, name):
        super(ImageDataset_Train, self).__init__()
        self.img_path_label = pd.read_csv(csv_file)
        self.transform = owntransforms
        self.name = name

    def __len__(self):
        return len(self.img_path_label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.img_path_label.iloc[idx, 0]

        if img_path != 'img_path':
            img = Image.open(img_path)
            img = self.transform(img)
            label = np.array(self.img_path_label.iloc[idx, 1])

        return {'image': img, 'label': label}


class ImageDataset_Test(Dataset):
    # def __init__(self, csv_file, img_size, filter_size, test_set):
    def __init__(self, csv_file, owntransforms, test_set= 'ff++'):#attribute,#
        self.transform = owntransforms
        self.img = []
        self.label = []


        with open(csv_file, newline='') as csvfile:
            rows = csv.reader(csvfile, delimiter=',')
            line_count = 0
            for row in rows:
                if line_count == 0:
                    line_count += 1
                    continue
                else:
                    if test_set == 'ff++':
                        img_path = row[0]
                        if img_path != 'img_path':
                            mylabel = int(row[1])
                            self.img.append(img_path)
                            self.label.append(mylabel)


                    if test_set == 'dfd':
                        img_path = row[0]
                        if img_path != 'img_path':
                            mylabel = int(row[1])
                            self.img.append(img_path)
                            self.label.append(mylabel)


                    if test_set == 'celebdf':
                        img_path = row[0]
                        if img_path != 'img_path':
                            mylabel = int(row[1])
                            self.img.append(img_path)
                            self.label.append(mylabel)


                    if test_set == 'dfdc':
                        img_path = row[0]
                        if img_path != 'img_path':
                            mylabel = int(row[1])
                            self.img.append(img_path)
                            self.label.append(mylabel)


    def __getitem__(self, index):

        path = self.img[index % len(self.img)]

        img = Image.open(path)
        label = self.label[index % len(self.label)]
        img = self.transform(img)
        data_dict = {}
        data_dict['image'] = img
        data_dict['label'] = label

        return data_dict

    def __len__(self):
        return len(self.img)
