#-- By Ender_F_L --#

import torch
from torch import nn

class SimPointTrackNet(nn.Module):
    """
    输入固定N个锥桶点的坐标，通过融合全局上下文来预测每个点的分类。
    """
    def __init__(self, num_points=10, num_classes=3):
        super(SimPointTrackNet, self).__init__()
        
        self.num_points = num_points
        self.num_classes = num_classes
        point_feature_dim = 8
        global_feature_dim = 16

        # 逐点特征提取器
        self.point_feat_extractor = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, point_feature_dim)
        )

        # 处理聚合后的全局特征的MLP
        self.global_feat_mlp = nn.Sequential(
            nn.Linear(point_feature_dim, global_feature_dim),
            nn.ReLU(),
            nn.Linear(global_feature_dim, global_feature_dim)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(point_feature_dim + global_feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_classes)
        )

    def forward(self, x: torch.Tensor):
        # 输入形状(Batch_size, N, 2)
        
        # 为每个点提取独立的特征
        point_features = self.point_feat_extractor(x)
        
        # 聚合所有点的特征，生成全局特征
        """
            找出每个点最有代表性的特征最后返回一张一维的带有全局特征的向量
        """
        global_feature_aggregated, _ = torch.max(point_features, dim=1, keepdim=True)
        global_feature = self.global_feat_mlp(global_feature_aggregated)

        # 将全局特征拼回每个点
        global_feature_expanded = global_feature.expand(-1, self.num_points, -1)
        fused_features = torch.cat([point_features, global_feature_expanded], dim=2) # 拼接

        # 融合了全局信息的丰富特征，进行最终分类
        # 实际相当于以全局特征为条件重新审视每个点
        logits = self.classifier(fused_features)
        
        return logits
    
    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        """
        用于推理和部署
        """
        if self.training:
            self.eval() 
        logits = self.forward(x)
        outputs = torch.softmax(logits, dim=-1)
        
        return outputs