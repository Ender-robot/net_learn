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

        targets, labels = self.sort(raw_targets, raw_labels)
        targets, labels = self.fixed_sampler(targets, labels, self.sample_num)

        dataset = {
            "targets" : torch.tensor(targets, dtype = torch.float32),
            "labels"  : torch.tensor(labels, dtype = torch.long),
            "name"    : self.data_file_list[idx]
        }

        return dataset
    
    # 重排序
    def sort(self, targets, labels):
        """
        使用贪心最近邻算法将点排序成一条路径。
        """

        # 从最左上角开始 (Y最大, 然后X最大)
        start_idx = np.lexsort((-targets[:, 0], -targets[:, 1]))[0]

        # 将点和标签转换为列表，方便移除操作
        remaining_targets = list(targets)
        remaining_labels = list(labels)
        
        start_point = remaining_targets.pop(start_idx)
        start_label = remaining_labels.pop(start_idx)
        
        sorted_targets = [start_point]
        sorted_labels = [start_label]
        
        current_point = start_point
        
        # 寻找最近邻
        while remaining_targets:
            distances_sq = np.sum((np.array(remaining_targets) - current_point)**2, axis=1)
            closest_idx = np.argmin(distances_sq)
            
            # 将找到的最近点加入排序列表，并从待选列表中移除
            current_point = remaining_targets.pop(closest_idx)
            current_label = remaining_labels.pop(closest_idx)
            
            sorted_targets.append(current_point)
            sorted_labels.append(current_label)
            
        return np.array(sorted_targets), np.array(sorted_labels)

    # 固定采样
    def fixed_sampler(self, targets, labels, num):

        # 当前样本数等于目标数，无需操作
        if len(labels) == num:
            return targets, labels

        # 当前样本数大于目标数
        elif len(labels) > num:
            # 直接截取序列的前num个点，相当于从远端删去
            sampled_targets = targets[:num]
            sampled_labels = labels[:num]
            return sampled_targets, sampled_labels

        # 当前样本数小于目标数
        else:
            num_to_add = num - len(labels)
            
            # 获取序列的最后一个点和标签
            last_target = targets[-1]
            last_label = labels[-1]
            
            # 重复添加最后一个点
            targets_to_add = np.tile(last_target, (num_to_add, 1))
            labels_to_add = np.tile(last_label, num_to_add)
            
            # 拼接
            sampled_targets = np.concatenate([targets, targets_to_add], axis=0)
            sampled_labels = np.concatenate([labels, labels_to_add], axis=0)
            
            return sampled_targets, sampled_labels
