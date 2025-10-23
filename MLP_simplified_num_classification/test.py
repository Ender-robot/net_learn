#-- By Ender_F_L --#

import torch
import torch.nn as nn
from torchvision.io import read_image
import argparse

from model.model import MLPSimplifiedNumClassification

def main():
    # 初始化一些参数
    parser = argparse.ArgumentParser(description = '推理参数')

    parser.add_argument('--device', type = int, help = '指定设备', default = "cpu")
    parser.add_argument('--img_file', type = str, help = '输入图片绝对路径')

    args = parser.parse_args() # 解析命令行参数

    # 训练时的数据处理方法和推理时应当保持一至
    img_tensor = read_image(args.img_file)
    img_tensor = img_tensor.float() / 255.0 # 归一化
    img_tensor = img_tensor.unsqueeze(0) # 在0位置再套一个维度，兼容batch维度，现在的形状是 1*3*100*100

    # 移动到指定设备
    if torch.cuda.is_available():
        device = torch.device("cuda", args.device)
        print(f"Used CUDA: {args.device}")
    else:
        device = torch.device("cpu")
        print(f"Used CPU")

    model = MLPSimplifiedNumClassification().to(device)
    img_tensor = img_tensor.to(device)

    # 推理
    # *将模型调整为评估模式* #
    model.eval()

    # 关闭梯度自动计算，现在不需要
    with torch.no_grad():
        outputs = model(img_tensor) # 模型最后一层的原始输出

    outputs = torch.nn.functional.softmax(outputs, dim=1) # 输出的形状为1*11，现在对11所在的维度进行激活，获得概率分布

    print(outputs)
    print("懒得写自动查找标签映射了，手动找一下吧......")

if __name__ == "__main__":
    main()
