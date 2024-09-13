## 项目目录结构如下
```
Evaluation:
    1chargeEM.py    用于对案由进行精确匹配
    2judgementF1.py     用于对判决结果进行f1评分
    3elementROUGE.py  用于对说理元素进行ROUGE评分
    4senmaticLLM.py     用于使用LLM对说理进行打分
    5ehticLegalF1.py   用于对伦理法理进行macro-f1评分
    6calculateFINAL.py  用于加权计算最终的结果
    eval.sh             运行的脚本
    __pycache__         
    result.txt         结果，其中是各个指标的评分
    testLLM.sh          测试LLM的正常运行的脚本
    final.txt           最终结果的文件
    README.md           
    testLLMfortest.py   测试LLM正常运行的python程序
```
## eval.sh的内容
指定好模型路径，数据路径，预测结果路径即可一键运行
```
#!/bin/bash
res_path="/home/sunminhao/CAIL2023/ssrd/mybaseline/eval/finalresult4/pred.json"
data_path="/home/sunminhao/CAIL2023/ssrd/mybaseline/dataproc/rawdata1"
file_path="/home/sunminhao/CAIL2023/ssrd/mybaseline/Evaluation/result.txt"
model_path="/home/data/public/DISC-LAWLLM"
python -m 1chargeEM --res_path "$res_path" --data_path "$data_path"
python -m 2judgementF1 --res_path "$res_path" --data_path "$data_path"
python -m 3elementROUGE --res_path "$res_path" --data_path "$data_path"
python -m 4senmaticLLM --res_path "$res_path" --data_path "$data_path" --model_path "$model_path"
python -m 5ehticLegalF1 --res_path "$res_path" --data_path "$data_path"
python -m 6calculateFINAL --file_path "$file_path"
```

# 评测采用的LLM
https://github.com/FudanDISC/DISC-LawLLM?tab=readme-ov-file#%E6%A6%82%E8%BF%B0