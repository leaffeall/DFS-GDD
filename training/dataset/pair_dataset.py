'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30

The code is designed for scenarios such as disentanglement-based methods where it is necessary to ensure an equal number of positive and negative samples.
'''

import torch
import random
import numpy as np
import csv
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import pickle
import os
import pandas as pd
from PIL import Image
import random


class pairDataset(Dataset):
    def __init__(self, csv_fake_file, csv_real_file, owntransforms):

        # Get real and fake image lists
        # Fix the label of real images to be 0 and fake images to be 1
        self.fake_image_list = pd.read_csv(csv_fake_file)
        self.real_image_list = pd.read_csv(csv_real_file)
        self.transform = owntransforms



    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        fake_img_path = self.fake_image_list.iloc[idx, 0]
        real_idx = random.randint(0, len(self.real_image_list) - 1)
        real_img_path = self.real_image_list.iloc[real_idx, 0]

        if fake_img_path != 'img_path':
            fake_img = Image.open(fake_img_path)
            fake_trans = self.transform(fake_img)
            fake_label = np.array(self.fake_image_list.iloc[idx, 1])

            fake_uni_label = np.array(self.fake_image_list.iloc[idx, 7])

        if real_img_path != 'img_path':
            real_img = Image.open(real_img_path)
            real_trans = self.transform(real_img)
            real_label = np.array(self.real_image_list.iloc[real_idx, 1])
            real_uni_label = np.array(self.real_image_list.iloc[real_idx, 1])
            # real_fair_label = np.array(self.real_image_list.iloc[real_idx, 2])

        return {"fake": (fake_trans, fake_label, fake_uni_label),
                "real": (real_trans, real_label, real_uni_label)}

    def __len__(self):
        return len(self.fake_image_list)

    @staticmethod
    def collate_fn(batch):
        """
        Collate a batch of data points.

        Args:
            batch (list): A list of tuples containing the image tensor, the label tensor,
                        the landmark tensor, and the mask tensor.

        Returns:
            A tuple containing the image tensor, the label tensor, the landmark tensor,
            and the mask tensor.
        """
        # Separate the image, label,  tensors for fake and real data
        fake_images, fake_labels, fake_uni_labels = zip(
            *[data["fake"] for data in batch])
        # print(fake_labels)
        fake_labels = tuple(x.item() for x in fake_labels)
        fake_uni_labels = tuple(x.item() for x in fake_uni_labels)
        real_images, real_labels, real_uni_labels = zip(
            *[data["real"] for data in batch])
        real_labels = tuple(x.item() for x in real_labels)
        real_uni_labels = tuple(x.item() for x in real_uni_labels)

        # Stack the image, label, tensors for fake and real data
        fake_images = torch.stack(fake_images, dim=0)
        fake_labels = torch.LongTensor(fake_labels)
        fake_uni_labels = torch.LongTensor(fake_uni_labels)

        real_images = torch.stack(real_images, dim=0)
        real_labels = torch.LongTensor(real_labels)
        real_uni_labels = torch.LongTensor(real_uni_labels)

        # Combine the fake and real tensors and create a dictionary of the tensors
        images = torch.cat([real_images, fake_images], dim=0)
        labels = torch.cat([real_labels, fake_labels], dim=0)
        uni_labels = torch.cat([real_uni_labels, fake_uni_labels], dim=0)


        data_dict = {
            'image': images,
            'label': labels,
            'label_uni': uni_labels
        }
        return data_dict
