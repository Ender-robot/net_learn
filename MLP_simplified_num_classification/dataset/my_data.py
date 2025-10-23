#-- By Ender_F_L --#

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image

import os
import json

# 构建自己的数据集格式
class MyDataset(Dataset):
    def __init__(self, imgs_dir, labels_dir, config_file):
        self.imgs_dir = imgs_dir
        self.labels_dir = labels_dir
        self.config_file = config_file

        # 检查和统计数据集数量
        self.img_files = os.listdir(self.imgs_dir)
        self.label_files = os.listdir(self.labels_dir)
        nums_imgs = len(self.img_files)
        nums_labels = len(self.label_files)

        # 读取标签编码表
        with open(config_file, 'r', encoding='utf-8') as file:
            self.config = json.load(file)

        if nums_imgs != nums_labels:
            self.data_nums = min(nums_imgs, nums_labels)
        else:
            self.data_nums = nums_imgs

    # 返回数据集大小
    def __len__(self):
        return self.data_nums
    
    # 根据索引idx，加载并返回一个样本
    def __getitem__(self, idx):
        img_path = os.path.join(self.imgs_dir, self.img_files[idx]) # 加载图像路径
        label_path = os.path.join(self.labels_dir, self.label_files[idx]) # 加载标签路径

        img = self.image2tensor(img_path) # 张量
        label_ = self.read_from_json(label_path) # 字符
        label = self.str2tensor(label_, self.config) # 张量

        return img, label

    # 将图像转化为tensor并归一化
    def image2tensor(self, image_file : str):
        img_tensor_raw = read_image(image_file) # 读取图片并转换为tensor.int8的形式
        img_tensor = img_tensor_raw.float() / 255.0 # 转换到tensor.float的形式并进行归一化

        return img_tensor
    
    # 从json文件中读取标签
    def read_from_json(self, label_file : str, key : str = "label"):
        with open(label_file, 'r', encoding='utf-8') as file:
            data = json.load(file)

        return data[key]
    
    # 将python的字符转换成tensor并归一化
    def str2tensor(self, string : str, config : dict):
        # 获取字符到整型的映射
        label_map = config["label_encode"]
        # 映射
        label_code = label_map[string]
        # 转成张量
        label_tensor = torch.tensor(label_code, dtype=torch.long)

        return label_tensor
