#-- By Ender_F_L --#

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import os
import argparse

from dataset.my_data import MyDataset
from model.model import MLPSimplifiedNumClassification

def main():
    # 初始化一些参数
    parser = argparse.ArgumentParser(description = '训练参数')

    parser.add_argument('--device', type = int, help = '指定设备', default = "cpu")
    parser.add_argument('--data_dir', type = str, help = '数据集根目录')
    parser.add_argument('--pt_dir', type = str, help = '模型权重目录')
    parser.add_argument('--epoch', type = int, help = '训练轮数')
    parser.add_argument('--batch_size', type = int, help = '批大小')
    parser.add_argument('--lr', type = float, help = '学习率', default = 0.001)

    args = parser.parse_args() # 解析命令行参数

    if torch.cuda.is_available():
        device = torch.device("cuda", args.device)
        print(f"Used CUDA: {args.device}")
    else:
        device = torch.device("cpu")
        print(f"Used CPU")

    # 实例化模型
    model = MLPSimplifiedNumClassification().to(device) # 实例化模型

    # 构建损失函数
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失，适用于多分类问题

    # 构建优化器
    # lr是学习率
    optimizer = optim.Adam(model.parameters(), lr = args.lr) # 将模型参数导入优化器，由优化器依照梯度指导参数更新

    #*** 训练模式 ***#
    model.train() # 指定模型为训练模式

    # 训练数据集路径
    train_imgs_dir = os.path.join(args.data_dir, r"images/train")
    train_labels_dir = os.path.join(args.data_dir, r"labels/train")
    dataset_config_file = os.path.join(args.data_dir, r"config.json")
    train_dataset = MyDataset(train_imgs_dir, train_labels_dir, dataset_config_file) # 初始化自定义的数据集类
    # 数据集加载器
    train_loader = DataLoader(dataset = train_dataset,
                              batch_size = args.batch_size,
                              shuffle = True
                              )
    num_epoch = args.epoch # 训练轮数

    for epoch in range(num_epoch):
        runtime_loss = 0.0 # 初始化运行时平均损失用于分析
        for i, (img, label) in enumerate(train_loader):
            # 将数据转移至指定设备
            img = img.to(device)
            label = label.to(device)

            # 前向传播
            outputs = model(img)

            # 计算损失
            loss = criterion(outputs, label)

            # 反向传播
            optimizer.zero_grad() # 清理就梯度
            loss.backward() # 求导计算梯度
            optimizer.step() # 更新权重

            runtime_loss += loss.item()

            if (i + 1) % 10 == 0:
                print(f"Epoch:{epoch + 1} / {num_epoch}  Step:{i + 1} / {len(train_loader)}  Loss:{(runtime_loss/10.0):.4f}")
                runtime_loss = 0.0

    pt_file = os.path.join(args.pt_dir, "weight.pt")
    torch.save(model.state_dict(), pt_file)
    print(f"Weights saved to {pt_file}")
    print("Finished Training !")

if __name__ == "__main__":
    main()
