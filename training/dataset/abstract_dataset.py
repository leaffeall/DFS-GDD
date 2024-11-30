# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30
# description: Abstract Base Class for all types of deepfake datasets.

from dataset.albu import IsotropicResize
import albumentations as A
from torchvision import transforms as T
from torch.utils import data
from torch.autograd import Variable
import torch
from collections import defaultdict
from PIL import Image
import random
import cv2
from copy import deepcopy
import numpy as np
import json
import glob
import yaml
import math
import os
import sys
sys.path.append('.')


class DeepfakeAbstractBaseDataset(data.Dataset):
    """
    Abstract base class for all deepfake datasets.
    """

    def __init__(self, config=None, mode='train'):
        """Initializes the dataset object.

        Args:
            config (dict): A dictionary containing configuration parameters.
            mode (str): A string indicating the mode (train or test).

        Raises:
            NotImplementedError: If mode is not train or test.
        """

        # Set the configuration and mode
        self.config = config
        self.mode = mode
        self.compression = config['compression']
        self.frame_num = config['frame_num'][mode]
        self.genimg_path = '/mntcephfs/sec_dataset/GenImage/'

        # Dataset dictionary
        self.image_list = []
        self.label_list = []

        # Set the dataset dictionary based on the mode
        if mode == 'train':
            dataset_list = config['train_dataset']
        elif mode == 'test':
            dataset_list = config['test_dataset']
        else:
            raise NotImplementedError(
                'Only train and test modes are supported.')
        self.dataset_list = dataset_list

        # Collect image and label lists
        image_list, label_list = self.collect_img_and_label(dataset_list)
        self.image_list, self.label_list = image_list, label_list

        # Create a dictionary containing the image and label lists
        self.data_dict = {
            'image': self.image_list,
            'label': self.label_list,
        }

        self.transform = self.init_data_aug_method()

    def init_data_aug_method(self):
        trans = A.Compose([
            A.HorizontalFlip(p=self.config['data_aug']['flip_prob']),
            A.Rotate(limit=self.config['data_aug']['rotate_limit'],
                     p=self.config['data_aug']['rotate_prob']),
            A.GaussianBlur(
                blur_limit=self.config['data_aug']['blur_limit'], p=self.config['data_aug']['blur_prob']),
            A.OneOf([
                IsotropicResize(
                    max_side=self.config['resolution'], interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                IsotropicResize(
                    max_side=self.config['resolution'], interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
                IsotropicResize(
                    max_side=self.config['resolution'], interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
            ], p=1),
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=self.config['data_aug']['brightness_limit'], contrast_limit=self.config['data_aug']['contrast_limit']),
                A.FancyPCA(),
                A.HueSaturationValue()
            ], p=0.5),
            A.ImageCompression(quality_lower=self.config['data_aug']['quality_lower'],
                               quality_upper=self.config['data_aug']['quality_upper'], p=0.5)
        ],
            keypoint_params=A.KeypointParams(
                format='xy') if self.config['with_landmark'] else None
        )
        return trans

    def collect_img_and_label(self, dataset_list):
        """Collects image and label lists.

        Args:
            dataset_list (dict): A dictionary containing dataset information.

        Returns:
            list: A list of image paths.
            list: A list of labels.

        Raises:
            ValueError: If image paths or labels are not found.
            NotImplementedError: If the dataset is not implemented yet.
        """
        # Initialize the label and frame path lists
        label_list = []
        frame_path_list = []
        landmarks_list = []
        masks_list = []

        # If the dataset dictionary is not empty, collect the image, label, landmark, and mask lists
        if dataset_list:
            # Iterate over the datasets in the dictionary
            for dataset_name in dataset_list:
                # Try to get the dataset information from the JSON file
                try:
                    with open(os.path.join(self.config['dataset_json_folder'], dataset_name + '.json'), 'r') as f:
                        dataset_info = json.load(f)
                except:
                    # If the JSON file does not exist, load the images from the genimage path
                    # ALL: ADM  BigGAN  glide  Midjourney  stable_diffusion_v_1_4  stable_diffusion_v_1_5  VQDM  wukong
                    dataset_path = os.path.join(self.genimg_path, dataset_name)
                    print(dataset_path, 'dataset_path!!!!!')
                    if os.path.exists(dataset_path):
                        if self.mode in ["val", "test"]:
                            mode = "val"
                        else:
                            mode = "train"

                        mode_path = os.path.join(dataset_path, mode)
                        if os.path.exists(mode_path):
                            for subfolder in ['nature', 'ai']:
                                subfolder_path = os.path.join(
                                    mode_path, subfolder)
                                if os.path.exists(subfolder_path):
                                    for img_name in os.listdir(subfolder_path):
                                        frame_path_list.append(
                                            os.path.join(subfolder_path, img_name))
                                        label = 0 if subfolder == 'nature' else 1
                                        label_list.append(label)
                    else:
                        print(
                            f'dataset {dataset_name} not exist! Lets skip it.')
                        continue  # skip if the dataset does not exist in the default image path

                # If JSON file exists, the following processes the data according to your original code
                else:
                    # FIXME: ugly, need to be modified here.
                    cp = None
                    if dataset_name == 'FaceForensics++_c40':
                        dataset_name = 'FaceForensics++'
                        cp = 'c40'
                    elif dataset_name == 'FF-DF_c40':
                        dataset_name = 'FF-DF'
                        cp = 'c40'
                    elif dataset_name == 'FF-F2F_c40':
                        dataset_name = 'FF-F2F'
                        cp = 'c40'
                    elif dataset_name == 'FF-FS_c40':
                        dataset_name = 'FF-FS'
                        cp = 'c40'
                    elif dataset_name == 'FF-NT_c40':
                        dataset_name = 'FF-NT'
                        cp = 'c40'
                    # Get the information for the current dataset
                    for label in dataset_info[dataset_name]:
                        sub_dataset_info = dataset_info[dataset_name][label][self.mode]
                        # Special case for FaceForensics++ and DeepFakeDetection, choose the compression type
                        if cp == None and dataset_name in ['FF-DF', 'FF-F2F', 'FF-FS', 'FF-NT', 'FaceForensics++', 'DeepFakeDetection', 'FaceShifter']:
                            sub_dataset_info = sub_dataset_info[self.compression]
                        elif cp == 'c40' and dataset_name in ['FF-DF', 'FF-F2F', 'FF-FS', 'FF-NT', 'FaceForensics++', 'DeepFakeDetection', 'FaceShifter']:
                            sub_dataset_info = sub_dataset_info['c40']
                        # Iterate over the videos in the dataset
                        for video_name, video_info in sub_dataset_info.items():
                            # Get the label and frame paths for the current video
                            if video_info['label'] not in self.config['label_dict']:
                                raise ValueError(
                                    f'Label {video_info["label"]} is not found in the configuration file.')
                            label = self.config['label_dict'][video_info['label']]
                            frame_paths = video_info['frames']

                            # Append the label and frame paths to the lists
                            label_list.extend([label]*len(frame_paths))
                            frame_path_list.extend(frame_paths)

                # Shuffle the label and frame path lists in the same order
                shuffled = list(zip(label_list, frame_path_list))
                random.shuffle(shuffled)
                label_list, frame_path_list = zip(*shuffled)

                return frame_path_list, label_list

        else:
            raise ValueError('No dataset is given.')

    def load_rgb(self, file_path):
        """
        Load an RGB image from a file path and resize it to a specified resolution.

        Args:
            file_path: A string indicating the path to the image file.

        Returns:
            An Image object containing the loaded and resized image.

        Raises:
            ValueError: If the loaded image is None.
        """
        size = self.config['resolution']
        assert os.path.exists(file_path), f"{file_path} does not exist"
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError('Loaded image is None: {}'.format(file_path))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
        return Image.fromarray(np.array(img, dtype=np.uint8))

    def load_mask(self, file_path):
        """
        Load a binary mask image from a file path and resize it to a specified resolution.

        Args:
            file_path: A string indicating the path to the mask file.

        Returns:
            A numpy array containing the loaded and resized mask.

        Raises:
            None.
        """
        size = self.config['resolution']
        if file_path is None:
            return np.zeros((size, size, 1))
        if os.path.exists(file_path):
            mask = cv2.imread(file_path, 0)
            if mask is None:
                mask = np.zeros((size, size))
            mask = cv2.resize(mask, (size, size))/255
            mask = np.expand_dims(mask, axis=2)
            return np.float32(mask)
        else:
            return np.zeros((size, size, 1))

    def load_landmark(self, file_path):
        """
        Load 2D facial landmarks from a file path.

        Args:
            file_path: A string indicating the path to the landmark file.

        Returns:
            A numpy array containing the loaded landmarks.

        Raises:
            None.
        """
        if file_path is None:
            return np.zeros((81, 2))
        if os.path.exists(file_path):
            landmark = np.load(file_path)
            return np.float32(landmark)
        else:
            return np.zeros((81, 2))

    def to_tensor(self, img):
        """
        Convert an image to a PyTorch tensor.
        """
        return T.ToTensor()(img)

    def normalize(self, img):
        """
        Normalize an image.
        """
        mean = self.config['mean']
        std = self.config['std']
        normalize = T.Normalize(mean=mean, std=std)
        return normalize(img)

    def data_aug(self, img, landmark=None, mask=None):
        """
        Apply data augmentation to an image, landmark, and mask.

        Args:
            img: An Image object containing the image to be augmented.
            landmark: A numpy array containing the 2D facial landmarks to be augmented.
            mask: A numpy array containing the binary mask to be augmented.

        Returns:
            The augmented image, landmark, and mask.
        """

        # Create a dictionary of arguments
        kwargs = {'image': img}

        # Check if the landmark and mask are not None
        if landmark is not None:
            kwargs['keypoints'] = landmark
            kwargs['keypoint_params'] = A.KeypointParams(format='xy')
        if mask is not None:
            kwargs['mask'] = mask

        # Apply data augmentation
        transformed = self.transform(**kwargs)

        # Get the augmented image, landmark, and mask
        augmented_img = transformed['image']
        augmented_landmark = transformed.get('keypoints')
        augmented_mask = transformed.get('mask')

        # Convert the augmented landmark to a numpy array
        if augmented_landmark is not None:
            augmented_landmark = np.array(augmented_landmark)

        return augmented_img, augmented_landmark, augmented_mask

    def __getitem__(self, index):
        """
        Returns the data point at the given index.

        Args:
            index (int): The index of the data point.

        Returns:
            A tuple containing the image tensor, the label tensor, the landmark tensor,
            and the mask tensor.
        """
        # Get the image paths and label
        image_path = self.data_dict['image'][index]
        label = self.data_dict['label'][index]

        # Get the mask and landmark paths
        mask_path = image_path.replace('frames', 'masks')  # Use .png for mask
        landmark_path = image_path.replace('frames', 'landmarks').replace(
            '.png', '.npy')  # Use .npy for landmark

        # Load the image
        try:
            image = self.load_rgb(image_path)
        except Exception as e:
            # Skip this image and return the first one
            print(f"Error loading image at index {index}: {e}")
            return self.__getitem__(0)
        image = np.array(image)  # Convert to numpy array for data augmentation

        # Load mask and landmark (if needed)
        if self.config['with_mask']:
            mask = self.load_mask(mask_path)
        else:
            mask = None
        if self.config['with_landmark']:
            landmarks = self.load_landmark(landmark_path)
        else:
            landmarks = None

        # Do transforms
        if self.config['use_data_augmentation']:
            image_trans, landmarks_trans, mask_trans = self.data_aug(
                image, landmarks, mask)
        else:
            image_trans, landmarks_trans, mask_trans = deepcopy(
                image), deepcopy(landmarks), deepcopy(mask)

        # To tensor and normalize
        image_trans = self.normalize(self.to_tensor(image_trans))
        if self.config['with_landmark']:
            landmarks_trans = torch.from_numpy(landmarks)
        if self.config['with_mask']:
            mask_trans = torch.from_numpy(mask_trans)

        return image_trans, label, landmarks_trans, mask_trans

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
        # Separate the image, label, landmark, and mask tensors
        images, labels, landmarks, masks = zip(*batch)

        # Stack the image, label, landmark, and mask tensors
        images = torch.stack(images, dim=0)
        labels = torch.LongTensor(labels)

        # Special case for landmarks and masks if they are None
        if landmarks[0] is not None:
            landmarks = torch.stack(landmarks, dim=0)
        else:
            landmarks = None

        if masks[0] is not None:
            masks = torch.stack(masks, dim=0)
        else:
            masks = None

        # Create a dictionary of the tensors
        data_dict = {}
        data_dict['image'] = images
        data_dict['label'] = labels
        data_dict['landmark'] = landmarks
        data_dict['mask'] = masks
        return data_dict

    def __len__(self):
        """
        Return the length of the dataset.

        Args:
            None.

        Returns:
            An integer indicating the length of the dataset.

        Raises:
            AssertionError: If the number of images and labels in the dataset are not equal.
        """
        assert len(self.image_list) == len(
            self.label_list), 'Number of images and labels are not equal'
        return len(self.image_list)


if __name__ == "__main__":
    with open('/home/zhiyuanyan/disfin/deepfake_benchmark/training/config/detector/xception.yaml', 'r') as f:
        config = yaml.safe_load(f)
    train_set = DeepfakeAbstractBaseDataset(
        config=config,
        mode='train',
    )
    train_data_loader = \
        torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=config['train_batchSize'],
            shuffle=True,
            num_workers=int(config['workers']),
            collate_fn=train_set.collate_fn,
        )
    from tqdm import tqdm
    for iteration, batch in enumerate(tqdm(train_data_loader)):
        # print(iteration)
        ...
        # if iteration > 10:
        #     break
