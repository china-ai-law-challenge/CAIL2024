# 多人多罪判决预测

## Updates
复赛阶段延长至**11月22日**结束。

封测提交说明：

- 提交数据说明：
    1. 提交结果格式同初复赛格式；
    2. 提交结果集对应的**模型ZIP文件**，ZIP文件中包括所有代码（数据处理、模型、测试等）、模型权重、模型使用说明文档（写清如何调用该模型），注：提交的代码应该是完整可运行代码。
- 封测阶段最多提交三次，且**无法直接查看结果分数或错误输出**。请选手们仔细检查后再提交。

---
初赛阶段延长至**10月31日**结束。


最终成绩分数构成：初赛成绩 * 20% + 复赛成绩 * 40% + 封测成绩 * 40%

## 任务介绍
该赛道由**南京大学软件学院语言智能处理研究组**承办。

刑事案件包含单被告人案件和多被告人案件。基于裁判文书中的案件事实描述推理单被告人的判决结果任务已多有研究，但针对多被告人案件的预测仍为少数。多被告人案件的判决需要从复杂的事实描述中精准把握并分析每位被告人相关的犯罪事实，明确被告人在整起案件中承担的角色，从而为每位被告人都做出合理的判决结果预测。

该任务旨在使用人工智能模型，根据判决书中包含多个被告人的案件事实描述，生成案件涉及所有被告人相应的判决结果预测。

## 数据介绍
数据集格式参见CAIL官网，文件为jsonlines格式。格式示例参见`data_example.jsonl`。

## 评估指标分数计算

关于本赛道分数的定义，欢迎阅读我们的论文[CMDL: A Large-Scale Chinese Multi-Defendant Legal Judgment Prediction Dataset](https://aclanthology.org/2024.findings-acl.351/)中的 `4.2.1 Case-level Evaluation Metrics`获取更详细的说明。
（P.S. 比赛原因，暂时关闭CMDL数据集开源下载通道🤗）

在 `case_level_metrics.py`中提供了从one-hot形式的prediction和label，加上被告人数量数组来计算case-level metrics的代码，方便在模型训练和调试过程中计算分数。

本任务的评价方式分为三部分，分别为罪名预测结果评价、法条预测结果评价及刑期预测结果评价。

1. 罪名预测结果评价说明:

- 给定案件 $c$ ，案件中共有 $n$ 个被告人。假设被告人 $d_i$ 犯了 $m_1$ 个罪名，模型预测 $m_2$ 个罪名，其中 $m_3$ 个罪名预测正确。该被告人的Precision、Recall、F1值计算如下：

$$
P_c^{d_i}=\frac{m_3}{m_2}
$$

$$
R_c^{d_i}=\frac{m_3}{m_1}
$$

$$
F1_c^{d_i}=\frac{2* P_c^{d_i}*R_c^{d_i}}{ P_c^{d_i}+R_c^{d_i}}
$$

- 案件 $c$ 整体的Precision、Recall、F1值为：

$$
P_c =\frac{\sum_{i=1}^{n}{P_c^{d_i}}}{n}
$$

$$
R_c =\frac{\sum_{i=1}^{n}{R_c^{d_i}}}{n}
$$

$$
F1_c =\frac{\sum_{i=1}^{n}{F1_c^{d_i}}}{n}
$$

- 整个测试集的Precision、Recall、F1值为：

$$
P =\frac{\sum{w_cP_c}}{\sum w_c},w_c=log_2n
$$

$$
R =\frac{\sum{w_cR_c}}{\sum w_c},w_c=log_2n
$$

$$
F1 =\frac{\sum{w_cF1_c}}{\sum w_c},w_c=log_2n
$$

2. 法条预测结果评价同罪名预测。
3. 刑期预测结果评价说明：

- 刑期预测效果采用准确率Accuracy评价。给定案件 $c$ ，案件中共有 $n$ 个被告人，假设 $k$ 个被告人的刑期预测正确，则：

$$
Acc_c=\frac{k}{n}
$$

- 整个测试集的Accuracy值为：

$$
Acc =\frac{\sum{w_cAcc_c}}{\sum w_c},w_c=log_2n
$$

最后总的评价得分为三个子任务评价值加权和：

- 总得分 = 0.3 * 罪名预测F1值 + 0.3 * 法条预测F1值 + 0.4 * 刑期预测Accuracy值

最终评测时我们会使用测试脚本从你们生成的json结果文件进行分数的计算。

## 提交格式说明

| 元素名称  | 数据类型   | 元素说明             |
| --------- | ---------- | -------------------- |
| id        | int        | 数据项序号           |
| judgments | list[dict] | 该案件的判决结果预测 |

其中judgments中每一项的数据格式如下：

| 元素名称 | 数据类型  | 元素说明                                                     |
| -------- | --------- | ------------------------------------------------------------ |
| name     | str       | 被告人姓名                                                   |
| charges  | list[str] | 被告人所犯的所有罪名列表                                     |
| articles | list[str] | 被告人所触犯的所有法条列表，精确到“款”级别。如“347-4”表示第347条第4款 |
| penalty  | int       | 预测的被告人刑期结果类别                                     |

**请以jsonlines的格式输出结果文件！每一行为一个case的预测结果，格式示例如下：**

```json
{
    "id": 0,
    "judgments": [{
        "name": "A",
        "charges": ["x罪", "y罪"],
        "articles": ["xxx", "yyy"],
        "penalty": 7  // penalty class (0-14)
    }, {
        "name": "B",
        "charges": ["z罪"],
        "articles": ["zzz"],
        "penalty": 8
    }]
}
```
**注：这里为了清晰展示输出格式进行分行和缩进，提交时一个case为一行**


## Baseline复现
代码中使用预加载到本地的模型，推荐将[模型下载](https://huggingface.co/google-bert/bert-base-chinese)到本地后，在`main.py`中指定为你本地存储模型的路径。或者仅使用`"bert-base-chinese"`从huggingface上直接下载使用。

修改 `run.sh`，将其中的数据集地址修改为本地保存数据集的地址：

```bash
# ===== Set your dataset path here =====
train_data_path="first_stage_train_5000.jsonl"
# no specific validation set provided, split it yourself if you like
val_data_path=""  
test_data_path="first_stage_test_300.jsonl"
```

运行 `sh run.sh`即可复现baseline。

## Baseline分数
Baseline在初赛测试集上的分数如下：

| Charge_F1 | Article_F1 | Penalty_Acc | Final_Score |
| ---- | ---- | ---- | ---- |
| 0.1937 | 0.1722 | 0.2371 | 0.2046 |