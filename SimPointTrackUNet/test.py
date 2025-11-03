#-- By Ender_F_L --#

import torch
from torch.utils.data import DataLoader

import os
import argparse
import matplotlib.pyplot as plt

from dataset.SimPointTrackUNetDataset import SimPointTrackNetDataset
from model.SimPointTrackUNet import SimPointTrackUNet
from utils import process as pc

def get_args():
    # 初始化一些参数
    parser = argparse.ArgumentParser(description = '测试参数')

    parser.add_argument('--device', type = str, help = '指定设备', default = "cpu")
    parser.add_argument('--data_dir', type = str, help = '数据集根目录')
    parser.add_argument('--log_dir', type = str, help = '输出目录')

    args = parser.parse_args() # 解析命令行参数

    return args

def test(args):
    # 选择设备
    if torch.cuda.is_available() and args.device != "cpu":
        device = torch.device(f"cuda:{args.device}")
        print(f"Used CUDA: {args.device}")
    else:
        device = torch.device("cpu")
        print(f"Used CPU")

    # 实例化模型
    model = SimPointTrackUNet(15).to(device) # 并模型转移至指定设备
    model.load_state_dict(torch.load(os.path.join(args.log_dir, "weight.pt"), map_location=device))

    # *将模型调整为评估模式* #
    model.eval()

    # 准备测试集
    test_dataset = SimPointTrackNetDataset(args.data_dir, 15)
    # 数据集加载器
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    total_points = 0
    num_correct = 0
    history_accuracy = []
    for _, data in enumerate(test_loader):
        # 将数据转移至指定设备
        targets = data["targets"].to(device)
        labels = data["labels"].to(device)
        name = data["name"]

        # 推理
        outputs = model.predict(targets)

        # 后处理
        classes, _ = pc.scores_process(outputs.transpose(1, 2)) # 置信度筛查

        # 分析数据
        accuracy, batch_total_points, batch_num_correct = pc.calculate_error_metrics(classes, labels)
        total_points += batch_total_points
        num_correct += batch_num_correct
        history_accuracy.append(accuracy)

        # 揪出严重失败样本
        if accuracy < 0.8:
            print(f"样本 {name} 准确度: {accuracy}")
            # print(classes)
            # print(targets)

    # 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(history_accuracy, label='Each test dataset accuracy')
    plt.title('accuracy')
    plt.xlabel('dataset num')
    plt.ylabel('accuracy')
    plt.legend() # 显示图例
    plt.grid(True) # 显示背景网格
    save_path = os.path.join(args.log_dir, 'accuracy.png')
    plt.savefig(save_path)
    print(f"每个测试样本准确度保存在了  {save_path}")
    plt.close()

    # 打印
    total_accuracy = num_correct / total_points * 100
    print(f"总体精确度在 {total_accuracy}%")

def main():
    args = get_args()
    test(args)

if __name__ == "__main__":
    main()
