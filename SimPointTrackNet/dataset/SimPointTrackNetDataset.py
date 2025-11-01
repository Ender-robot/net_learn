#-- By Ender_F_L --#

import torch
from torch.utils.data import Dataset

import os
import numpy as np

# 构建自己的数据集格式
class SimPointTrackNetDataset(Dataset):
    def __init__(self, data_dir, sample_num = 10):
        self.data_dir = data_dir
        self.sample_num = sample_num

        # 检查和统计数据集数量
        self.data_file_list = os.listdir(self.data_dir)
        self.data_nums = len(self.data_file_list)

    # 返回数据集大小
    def __len__(self):
        return self.data_nums
    
    # 根据索引idx，加载并返回一个样本
    def __getitem__(self, idx): 
        data_path = os.path.join(self.data_dir, self.data_file_list[idx]) # 加载数据路径
        with np.load(data_path, allow_pickle=True) as data: # 加载一个数据
            raw_targets = data["targets"]
            raw_labels = data["labels"]

        targets, labels = self.fixed_sampler(raw_targets, raw_labels, self.sample_num)

        dataset = {
            "targets" : torch.tensor(targets, dtype = torch.float32),
            "labels"  : torch.tensor(labels, dtype = torch.long),
            "name"    : self.data_file_list[idx]
        }

        return dataset
    
    # 固定采样
    def fixed_sampler(self, targets, labels, num):

        # 当前样本数等于目标数，无需操作
        if len(labels) == num:
            return targets, labels

        # 当前样本数大于目标数
        elif len(labels) > num:
            
            # 所有点的质心
            centroid = np.mean(targets, axis=0)
            
            # 每个点到质心的欧氏距离的平方
            distances_sq = np.sum((targets - centroid)**2, axis=1)
            
            # 获取按距离从小到大排序的索引
            sorted_indices = np.argsort(distances_sq)
            
            # 保留距离最近的num个点的索引
            indices_to_keep = sorted_indices[:num]
            
            # 5. 根据索引筛选targets和labels
            sampled_targets = targets[indices_to_keep]
            sampled_labels = labels[indices_to_keep]
            
            return sampled_targets, sampled_labels

        # 当前样本数小于目标数
        else:
            
            num_to_add = num - len(labels)
            
            # 对索引进行重采样
            indices_to_add = np.random.choice(len(labels), size=num_to_add, replace=True)
            
            # 获取需要补充的targets和labels
            targets_to_add = targets[indices_to_add]
            labels_to_add = labels[indices_to_add]
            
            # 将新抽取的样本与原始样本拼接起来
            sampled_targets = np.concatenate([targets, targets_to_add], axis=0)
            sampled_labels = np.concatenate([labels, labels_to_add], axis=0)
            
            return sampled_targets, sampled_labels
