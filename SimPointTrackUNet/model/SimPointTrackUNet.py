#-- By Ender_F_L --#

import torch
from torch import nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
        卷积-归一-激活*2
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class SimPointTrackUNet(nn.Module):
    """
    一个用于点云轨迹分割的轻量级1D U-Net。
    输入的数据必须是经过排序的、代表路径的有序点序列。
    """
    def __init__(self, num_points=15, num_classes=3):
        super(SimPointTrackUNet, self).__init__()
        
        self.num_points = num_points
        self.num_classes = num_classes
        
        # 初始特征提取器
        self.initial_feat = nn.Linear(2, 8)
        
        # --- 编码器 ---
        # 负责提取特征并逐渐扩大感受野，理解宏观结构
        self.down1 = DoubleConv(8, 16)
        self.down2 = DoubleConv(16, 32)
        # MaxPool1d用于下采样，将序列长度减半
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # --- 瓶颈层 ---
        # 信息压缩最极致的地方，拥有最大的感受野
        self.bottleneck = DoubleConv(32, 64)
        
        # --- 解码器 ---
        # 负责将宏观特征与精细的局部特征融合，并恢复序列长度
        self.up1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # +64 来自down2的跳跃连接
        self.up1_conv = DoubleConv(64 + 32, 32) 
        
        self.up2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # +32 来自down1的跳跃连接
        self.up2_conv = DoubleConv(32 + 16, 16) 
        
        # --- 输出层 ---
        # 最后的1x1卷积，将每个点的特征向量映射到最终的类别分数上
        self.out_conv = nn.Conv1d(16, self.num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor):
        # 输入(Batch_size, N, 2)
        
        # 初始特征提取
        x1_feat = self.initial_feat(x) # (B, N, 8)
        
        # 转置为(Batch, Channels, Length)
        x1 = x1_feat.transpose(1, 2) # (B, 8, N)
        
        # --- 编码 ---
        d1 = self.down1(x1)   # (B, 16, 15)
        p1 = self.pool(d1)    # (B, 16, 7)
        
        d2 = self.down2(p1)   # (B, 32, 7)
        p2 = self.pool(d2)    # (B, 32, 3)
        
        # --- 瓶颈 ---
        b = self.bottleneck(p2) # (B, 64, 3)
        
        # --- 解码 ---
        u1 = self.up1_upsample(b)   # (B, 64, 6)
        # 处理池化导致尺寸不匹配的问题，使用插值来对齐
        if u1.shape[2] != d2.shape[2]:
            u1 = F.interpolate(u1, size=d2.shape[2], mode='nearest') # (B, 64, 7)
        
        # 跳跃连接，将解码器特征与编码器对应层的特征在通道维度上拼接
        c1 = torch.cat([u1, d2], dim=1) # (B, 64+32, 7)
        u1_conv = self.up1_conv(c1)    # (B, 32, 7)
        
        u2 = self.up2_upsample(u1_conv) # (B, 32, 14)
        # 处理池化导致尺寸不匹配的问题，使用插值来对齐
        if u2.shape[2] != d1.shape[2]:
            u2 = F.interpolate(u2, size=d1.shape[2], mode='nearest') # -> (B, 32, 15)
            
        c2 = torch.cat([u2, d1], dim=1) # (B, 32+16, 15)
        u2_conv = self.up2_conv(c2)    # (B, 16, 15)

        # --- 输出 ---
        # 输出(Batch, Classes, Length)
        logits = self.out_conv(u2_conv) # (B, 3, 15)
        
        return logits
    
    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        if self.training:
            self.eval() 
        logits = self.forward(x) # logits (B, C, N)
        outputs = torch.log_softmax(logits, dim=1)
        return outputs
    