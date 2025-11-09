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
from utils import process as pc

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

    # 数据集路径
    train_dataset = SimPointTrackNetDataset(os.path.join(args.data_dir, "train"), 15) # 初始化自定义的数据集类
    test_dataset = SimPointTrackNetDataset(os.path.join(args.data_dir, "test"), 15)
    # 数据集加载器
    train_loader = DataLoader(dataset = train_dataset,
                              batch_size = args.batch_size,
                              shuffle = True
                              )
    test_eval_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    train_eval_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False)

    num_epoch = args.epoch # 训练轮数

    loss_history = [] # 空数组用于绘图
    test_accuracy_history = []
    train_accuracy_history = []
    for epoch in range(num_epoch):

        #*** 训练模式 ***#
        model.train() # 指定模型为训练模式
        runtime_loss_sum = 0.0 # 初始化运行时平均损失用于分析
        step_cnt = 0
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

            # 记录
            runtime_loss_sum += loss.item()
            step_cnt = i + 1

        # 训练结果打印
        print(f"[Train] : Epoch:{epoch + 1} / {num_epoch}  Loss:{(runtime_loss_sum/step_cnt):.4f}")
        loss_history.append(runtime_loss_sum/step_cnt)

        #*** 评估模式 ***#
        model.eval()
        targets = data["targets"].to(device)
        labels = data["labels"].to(device)

        # 验证集评估
        test_total_points, test_num_correct = 0, 0
        for i, data in enumerate(test_eval_loader):
            # 将数据转移至指定设备
            targets = data["targets"].to(device)
            labels = data["labels"].to(device)

            # 推理
            outputs = model.predict(targets)
            
            # 后处理
            classes, _ = pc.scores_process(outputs.transpose(1, 2)) # 置信度筛查

            # 分析数据
            _, batch_total_points, batch_num_correct = pc.calculate_error_metrics(classes, labels)
            test_total_points += batch_total_points
            test_num_correct += batch_num_correct
            
        test_accuracy_history.append(test_num_correct/test_total_points)

        # 训练集评估
        train_total_points, train_num_correct = 0, 0
        for i, data in enumerate(train_eval_loader):
            # 将数据转移至指定设备
            targets = data["targets"].to(device)
            labels = data["labels"].to(device)

            # 推理
            outputs = model.predict(targets)
            
            # 后处理
            classes, _ = pc.scores_process(outputs.transpose(1, 2)) # 置信度筛查

            # 分析数据
            _, batch_total_points, batch_num_correct = pc.calculate_error_metrics(classes, labels)
            train_total_points += batch_total_points
            train_num_correct += batch_num_correct

        train_accuracy_history.append(train_num_correct/train_total_points)

        # 评估结果打印
        print(f"[Eval] : Epoch:{epoch + 1} / {num_epoch}  Test accuracy:{(test_num_correct/test_total_points):.4f}  Train accuracy:{(train_num_correct/train_total_points):.4f}")

        scheduler.step() # 每个轮结束后更新学习率

    # 绘图
    # 损失曲线
    plt.figure(figsize=(12, 6))
    plt.plot(loss_history, label='Average Loss per 5 Steps')
    plt.title('loss')
    plt.xlabel('epcho')
    plt.ylabel('loss')
    plt.legend() # 显示图例
    plt.grid(True) # 显示背景网格
    save_path = os.path.join(args.log_dir, 'loss.png')
    plt.savefig(save_path)
    print(f"Loss curve image saved to {save_path}")
    plt.close()

    # 验证曲线
    plt.figure(figsize=(12, 6))
    plt.plot(test_accuracy_history, label='test dataset accuracy')
    plt.plot(train_accuracy_history, label='train dataset accuracy')
    plt.title('accuracy')
    plt.xlabel('epcho')
    plt.ylabel('accuracy')
    plt.legend() # 显示图例
    plt.grid(True) # 显示背景网格
    save_path = os.path.join(args.log_dir, 'eval_accuracy.png')
    plt.savefig(save_path)
    print(f"Eval accuracy curve image saved to {save_path}")
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
