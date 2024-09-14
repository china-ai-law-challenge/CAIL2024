import math
import torch

# 这里提供一种基于 预测结果和标签的one-hot vector，及被告人数量数组 计算case-level指标的方法

def get_weight(defendant_num):
    """
        给定案件被告人人数，计算基础权重
    Args:
        defendant_num (_type_): 案件被告人数量
    Returns:
        float: 权重值，计算方式为log2(D), D为被告人数量
    """
    if isinstance(defendant_num, torch.Tensor):
        # defendant_num为tensor类型
        return math.log(defendant_num.item(), 2)
    else:
        # defendant_num为直接的数值类型
        return math.log(defendant_num, 2)


def get_per_defendant_metrics_weight(defendant_num):
    """
        计算最终结果中每个被告人对应的权重，是get_weight(defendant_num)/defendant_num
    Args:
        defendant_num (_type_): 案件被告人数量
    Returns:
        _type_: _description_
    """
    if isinstance(defendant_num, torch.Tensor):
        assert defendant_num.item() != 0
        return get_weight(defendant_num.item()) / defendant_num.item()
    else:
        assert defendant_num != 0
        return get_weight(defendant_num) / defendant_num


def get_weighted_per_defendant_metrics(outputs, labels, weight_tensor):
    """
        通过计算per-defendant指标计算per-case指标后进行加权
    Args:
        outputs (_type_): 模型预测
        labels (_type_): 真实标签
        weight_tensor (_type_): 这里的每个weight为get_weight(defendant_num)/defendant_num
    Returns:
        acc, precision, recall, f1: 加权后的最终指标
    """
    assert outputs.shape == labels.shape
    assert outputs.size(0) == weight_tensor.size(0)
    acc, precision, recall, f1 = 0, 0, 0, 0
    sample_num = outputs.size(0)
    # 加权平均的分母
    weight_total = torch.sum(weight_tensor).item()
    TPs = torch.sum(outputs * labels, dim=1)
    FPs = torch.sum(outputs * (1 - labels), dim=1)
    FNs = torch.sum((1 - outputs) * labels, dim=1)
    TNs = torch.sum((1 - outputs) * (1 - labels), dim=1)
    precisions = TPs / torch.clamp(TPs + FPs, min=1e-8)
    recalls = TPs / torch.clamp(TPs + FNs, min=1e-8)
    f1s = 2 * precisions * recalls / torch.clamp(precisions + recalls, min=1e-8)
    # exact match accuracy
    acc = torch.sum(weight_tensor * torch.all(outputs == labels, dim=1).float()) / weight_total
    precision = torch.sum(weight_tensor * precisions) / weight_total
    recall = torch.sum(weight_tensor * recalls) / weight_total
    f1 = torch.sum(weight_tensor * f1s) / weight_total
    return acc.item(), precision.item(), recall.item(), f1.item()


def get_case_level_metrics_by_per_defendant_metrics(outputs, labels, defendant_nums):
    """
        计算case-level metrics
    Args:
        outputs (torch.Tensor): size(batch_size, num_labels)，每个元素为0或1
        labels (torch.Tensor): size(batch_size, num_labels)，每个元素为0或1
        defendant_nums (List): len(batch_size) 每条输入对应案件的被告人数量数组
    Returns:
        三个任务的结果
    """
    # 对每个被告人计算accuracy, precision, recall, f1
    weight_tensor = torch.tensor(list(map(get_per_defendant_metrics_weight, defendant_nums))).to(outputs["charge"].device)
    charge_acc, charge_precision, charge_recall, charge_f1 = get_weighted_per_defendant_metrics(outputs["charge"], labels["charge"], weight_tensor)
    article_acc, article_precision, article_recall, article_f1 = get_weighted_per_defendant_metrics(outputs["article"], labels["article"], weight_tensor)
    penalty_acc, penalty_precision, penalty_recall, penalty_f1 = get_weighted_per_defendant_metrics(outputs["penalty"], labels["penalty"], weight_tensor)
    result = {
        "case_level_Acc_Acc": charge_acc,
        "case_level_Acc_Precision": charge_precision,
        "case_level_Acc_Recall": charge_recall,
        "case_level_Acc_F1": charge_f1,
        "case_level_Art_Acc": article_acc,
        "case_level_Art_Precision": article_precision,
        "case_level_Art_Recall": article_recall,
        "case_level_Art_F1": article_f1,
        "case_level_Pen_Acc": penalty_acc,
        "case_level_Pen_Precision": penalty_precision,
        "case_level_Pen_Recall": penalty_recall,
        "case_level_Pen_F1": penalty_f1
    }
    print(result)
    return result


def calculate_final_score(result):
    """
        计算最终得分
    Args:
        result (dict): 通过上述方法得到的三个任务的case-level结果；最终分数为
            0.3 * charge_f1 + 0.3 * article_f1 + 0.4 * penalty_acc
    Returns:
        float: 最终得分
    """
    # 三个任务的结果
    charge_f1 = result["case_level_Acc_F1"]
    article_f1 = result["case_level_Art_F1"]
    penalty_acc = result["case_level_Pen_Acc"]
    return 0.3 * charge_f1 + 0.3 * article_f1 + 0.4 * penalty_acc