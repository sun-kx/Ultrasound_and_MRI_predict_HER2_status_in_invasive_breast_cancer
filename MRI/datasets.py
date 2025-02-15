import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
from monai.transforms import Compose, ScaleIntensity, AddChannel, Resize, RandRotate90, EnsureType

def standard(data, mu, sigma):
    data = (data - mu) / sigma
    return data

def default_loader(path):
    img = sitk.GetArrayFromImage(sitk.ReadImage(path))
    return img

def resize_3D(img, size, method=cv2.INTER_LINEAR):
    img_ori = img.copy()
    for i in range(len(size)):
        size_obj = list(size).copy()
        size_obj[i] = img_ori.shape[i]
        img_new = np.zeros(size_obj)
        for j in range(img_ori.shape[i]):
            if i == 0:
                img_new[j, :, :] = cv2.resize(img_ori[j, :, :].astype('float'), (size[2], size[1]), interpolation=method)
            elif i == 1:
                img_new[:, j, :] = cv2.resize(img_ori[:, j, :].astype('float'), (size[2], size[0]), interpolation=method)
            else:
                img_new[:, :, j] = cv2.resize(img_ori[:, :, j].astype('float'), (size[1], size[0]), interpolation=method)
        img_ori = img_new.copy()
    return img_ori

class NeoadjuvantImmunotherapy_Dataset(Dataset):
    def __init__(self, root_dir, parameter, image_size=(128, 128, 128), transform=None, standard=standard, loader=default_loader, num_class=2):
        """
        root_dir: 文件夹路径，例如 'train/' 或 'test/'，其下有子文件夹 '0' 和 '1'
        parameter: 标准化所需的均值和方差（mu, sigma）
        image_size: 目标图像大小，默认为 (128, 128, 128)
        transform: 图像大小调整函数
        standard: 标准化函数
        loader: 加载图像函数，默认使用 SimpleITK
        num_class: 分类数量，默认为 2
        """
        self.imgs = []
        self.mu, self.sigma = parameter
        self.image_size = image_size
        self.transform = transform
        self.standard = standard
        self.loader = loader


        # 遍历文件夹中的图片，按文件夹名（0 或 1）设置标签
        for label in range(num_class):
            class_dir = os.path.join(root_dir, str(label))
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.imgs.append((img_path, label))

    def __getitem__(self, index):
        img_path, label = self.imgs[index]
        img = self.loader(img_path)
        # print(img)
        # img[img >= 1024] = 1024
        # img[img < -1024] = -1024

        # img = self.standard(img, self.mu, self.sigma)
        img = self.transform(img)
        print(img.shape)
        # print(img[np.newaxis,:].shape)
        # img = img[np.newaxis,:]
        # img = np.expand_dims(img, axis=0)  # 添加一个维度，变为 (1, 112, 112, 112)
        # img = np.repeat(img, 3, axis=0)  # 在第 0 个维度上重复 3 次，变为 (3, 112, 112, 112)
        return img, int(label)

    def __len__(self):
        return len(self.imgs)
