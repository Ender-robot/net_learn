#-- By Ender_F_L --#

import torch
import torch.nn as nn

# 定义一个三层感知机神经网络来进行数字识别
class MLPSimplifiedNumClassification():
    """
        定义一个简单的三层全连接神经网络，一个输入层，两个隐藏层，一个输出层，输入一张28*28的图片
        实现简单数字识别
    """

    def __init__(self):
        super(MLPSimplifiedNumClassification, self).__init__()

        self.fc1 = nn.Linear(100 * 100, 128) # 第一个全连接层，输入为28*28，输出为128
        self.fc2 = nn.Linear(128, 64) # 第二个全连接层，输入为128,输出为64
        self.fc3 = nn.Linear(64, 10) # 第三个全连接层，也是输出层，输入为64，输出为10，对应10个数字

    # 定义一次前向传播
    def forword(self, x:torch.Tensor):
        x = x.view() # 将输入的28*28的图像扁平化为一张一维数组
        x = torch.relu(self.fc1(x)) # 输入第一层神经网络并使用ReLu激活
        x = torch.relu(self.fc2(x)) # 输入第二层神经网络并使用ReLu激活
        x = self.fc3(x) # 第三层不需要激活因为损失函数期望一个原始的输出，自己内部会处理

        return x
    