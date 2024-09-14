import json
import jsonlines
from tqdm import tqdm

article2idx = {}
charge2idx = {}
with open("./data/article2idx_paragraphs.json") as f:
    article2idx = json.load(f)

with open("./data/charge2idx.json") as f:
    charge2idx = json.load(f)

charge_num = len(charge2idx)
article_num = len(article2idx)
penalty_num = 15
print("charge_num:", charge_num)
print("article_num:", article_num)

def get_penalty_label(penalty: dict):
    if penalty["death_penalty"]:
        # 死刑
        return 0
    elif penalty["life_imprisonment"]:
        # 无期徒刑
        return 1
    elif penalty["detention"] > 0:
        # 拘役
        return 2
    elif penalty["surveillance"] > 12:
        # 管制一年以上
        return 3
    elif penalty["surveillance"] > 0:
        # 管制一年以下
        return 4
    elif penalty["imprisonment"] > 120:
        # 有期徒刑10年以上
        return 5
    elif penalty["imprisonment"] > 84:
        # 有期徒刑7年以上，10年以下
        return 6
    elif penalty["imprisonment"] > 60:
        # 有期徒刑5年以上，7年以下
        return 7
    elif penalty["imprisonment"] > 36:
        # 有期徒刑3年以上，5年以下
        return 8
    elif penalty["imprisonment"] > 24:
        # 有期徒刑2年以上，3年以下
        return 9
    elif penalty["imprisonment"] > 12:
        # 有期徒刑1年以上，2年以下
        return 10
    elif penalty["imprisonment"] > 9:
        # 有期徒刑9个月以上，1年以下
        return 11
    elif penalty["imprisonment"] > 6:
        # 有期徒刑6个月以上，9个月以下
        return 12
    elif penalty["imprisonment"] > 0:
        # 有期徒刑6个月以下
        return 13
    else:
        # 免予刑事处罚
        return 14


def preprocess_data(data_path: str, data_num=None):
    """
        从数据集文件中读取数据，返回数据集的各个字段
    Args:
        data_path (str): 数据集路径
        data_num (int): 默认None，只加载前`data_num`个条目，否则加载全部
    Returns:
        facts: 案件事实数组（被告人姓名+每个被告人事实为基础事实）
        charge_labels: 各被告人罪名标签数组
        article_labels: 各被告人法条标签数组
        penalty_labels: 各被告人刑期标签数组
        defendant_nums: 每个案件对应的被告人数量数组
    """
    facts = []
    charge_labels = []
    article_labels = []
    penalty_labels = []
    defendant_nums = []
    cnter = 0
    with jsonlines.open(data_path, "r") as reader:
        for case in tqdm(reader):
            base_fact = case["fact"]
            for outcome in case["outcomes"]:
                # 一个outcome对应一个被告人
                defendant_fact = f"预测{outcome['name']}。" + base_fact
                defendant_fact = defendant_fact.replace("\n", "")
                charge_label = [0] * charge_num
                article_label = [0] * article_num
                penalty_label = [0] * penalty_num
                penalty_idx = get_penalty_label(outcome["penalty"])
                penalty_label[penalty_idx] = 1
                for charge in outcome["charges"]:
                    charge_idx = charge2idx[charge] if charge in charge2idx else -1
                    if charge_idx != -1:
                        charge_label[charge_idx] = 1
                for article in outcome["articles"]:
                    article_idx = article2idx[article] if article in article2idx else -1
                    if article_idx != -1:
                        article_label[article_idx] = 1

                facts.append(defendant_fact)
                charge_labels.append(charge_label)
                article_labels.append(article_label)
                penalty_labels.append(penalty_label)
            # each sample has a `defendant_num` of the corresponding case
            defendant_nums.extend([len(case["outcomes"])] * len(case["outcomes"]))
            cnter += 1
            if data_num is not None and cnter >= data_num:
                break

    return facts, charge_labels, article_labels, penalty_labels, defendant_nums
