#-- By Ender_F_L --#

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

import os
import argparse
import matplotlib.pyplot as plt

from dataset.SimPointTrackUNetDataset import SimPointTrackNetDataset
from model.SimPointTrackUNet import SimPointTrackUNet

def get_args():
    # 初始化一些参数
    parser = argparse.ArgumentParser(description = '训练参数')

    parser.add_argument('--device', type = str, help = '指定设备', default = "cpu")
    parser.add_argument('--data_dir', type = str, help = '数据集根目录')
    parser.add_argument('--log_dir', type = str, help = '输出目录')
    parser.add_argument('--epoch', type = int, help = '训练轮数')
    parser.add_argument('--batch_size', type = int, help = '批大小')
    parser.add_argument('--lr', type = float, help = '学习率', default = 0.001)

    args = parser.parse_args() # 解析命令行参数

    return args

def train(args):
    # 选择设备
    if torch.cuda.is_available() and args.device != "cpu":
        device = torch.device(f"cuda:{args.device}")
        print(f"Used CUDA: {args.device}")
    else:
        device = torch.device("cpu")
        print(f"Used CPU")

    # 实例化模型
    model = SimPointTrackUNet(15).to(device) # 并模型转移至指定设备

    # 构建损失函数
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失，适用于多分类问题

    # 构建优化器
    # lr是学习率
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4) # 将模型参数导入优化器，由优化器依照梯度指导参数更新

    # 学习率调度器，动态调整学习率
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)

    #*** 训练模式 ***#
    model.train() # 指定模型为训练模式

    # 训练数据集路径
    train_dataset = SimPointTrackNetDataset(args.data_dir, 15) # 初始化自定义的数据集类
    # 数据集加载器
    train_loader = DataLoader(dataset = train_dataset,
                              batch_size = args.batch_size,
                              shuffle = True
                              )
    num_epoch = args.epoch # 训练轮数

    loss_history = [] # 空数组用于绘图
    for epoch in range(num_epoch):
        runtime_loss_sum = 0.0 # 初始化运行时平均损失用于分析
        for i, data in enumerate(train_loader):

            # 将数据转移至指定设备
            targets = data["targets"].to(device)
            labels = data["labels"].to(device)

            # 前向传播
            outputs = model(targets)

            # 计算损失
            loss = criterion(outputs, labels)

            # 更新
            optimizer.zero_grad() # 清理旧梯度
            loss.backward() # 反向传播，求导计算梯度
            optimizer.step() # 更新权重

            # 日志
            runtime_loss_sum += loss.item()

            if (i + 1) % 5 == 0:
                print(f"Epoch:{epoch + 1} / {num_epoch}  Step:{i + 1} / {len(train_loader)}  Loss:{(runtime_loss_sum/5.0):.4f}")
                loss_history.append(runtime_loss_sum/5.0)
                runtime_loss_sum = 0.0

        scheduler.step() # 每个轮结束后更新学习率

    # 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(loss_history, label='Average Loss per 5 Steps')
    plt.title('loss')
    plt.xlabel('step(each 5)')
    plt.ylabel('loss')
    plt.legend() # 显示图例
    plt.grid(True) # 显示背景网格
    save_path = os.path.join(args.log_dir, 'loss.png')
    plt.savefig(save_path)
    print(f"Loss curve image saved to {save_path}")
    plt.close()

    # 打印
    pt_file = os.path.join(args.log_dir, "weight.pt")
    torch.save(model.state_dict(), pt_file)
    print(f"Weights saved to {pt_file}")
    print("Finished Training !")

def main():
    args = get_args()
    train(args)

if __name__ == "__main__":
    main()
