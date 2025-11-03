#-- By Ender_F_L --#

import torch

def scores_process(raw_outputs):
    """
        输入原始输入进行分数筛选
    """
    confidences, predicted_indices = torch.max(raw_outputs, dim=-1)

    # 移除批大小维度
    predicted_indices = predicted_indices.squeeze(0)
    confidences = confidences.squeeze(0) 

    return predicted_indices, confidences

def remove_sample_points(predicted_indices, confidences, targets):
    """
        Input:
            predicted_indices: 经过分数筛查后的标签索引（类别）
            confidences: （置信度分数）
            targets: 模型输入的经过固定采样的目标点
    """
    targets = targets.squeeze(0) 

    labels_reshaped = predicted_indices.float().unsqueeze(1)
    combined = torch.cat([targets, labels_reshaped], dim=1)

    unique_combined, inverse_indices, counts = torch.unique(
        combined, dim=0, sorted=True, return_inverse=True, return_counts=True
    )

    _, first_occurrence_indices = torch.unique(inverse_indices, sorted=True, return_inverse=True)

    unique_targets = unique_combined[:, :2]
    unique_labels = unique_combined[:, 2].long()
    
    unique_confidences = confidences[first_occurrence_indices]
    
    return unique_targets, unique_labels, unique_confidences

def calculate_error_metrics(predicted_labels, true_labels):
    """
    通过比较预测标签和真实标签，计算准确率和错误率。

    Args:
        predicted_labels (torch.Tensor): 一个批次经处理后的类别列表
        true_labels (torch.Tensor): 一个批次正确的类别列表（经过固定采样的）

    Returns: 正确率
    """
    # 逐元素比较，得到一个布尔张量
    correct_predictions = (predicted_labels == true_labels)

    # 计算总点数
    total_points = true_labels.numel()

    # 计算预测正确的点的数量
    num_correct = torch.sum(correct_predictions).item()

    # 计算准确率
    accuracy = num_correct / total_points

    return accuracy, total_points, num_correct
