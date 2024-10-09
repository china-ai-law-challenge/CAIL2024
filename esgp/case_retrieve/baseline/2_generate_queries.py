import json
from tqdm import tqdm
import jieba


file_path = 'workplace/CAIL_Task4/data/sample_data/new_queries_fact.json'
with open(file_path, 'r', encoding='utf-8') as file:
    queries = json.load(file)
    
qid_idx_writer = open('workplace/CAIL_Task4/data/sample_data/queries_fact.tsv','w',newline='')
cnt = 0
for query in queries:  
    cnt += 1  
    qid = query["id"]  
    # 截断yishen_fact字段到前512个字符  
    yishen = query['yishen_fact'].replace('\n', '').replace('\t', '')  
    yishen_fact_truncated = yishen[:512]  
    yishen_fact_cleaned = yishen_fact_truncated.replace('\n', '').replace('\t', '')  
    # 对清理后的文本进行分词
    words = jieba.lcut(yishen_fact_cleaned)
    # 使用制表符连接分词结果，形成查询文本  
    qry = " ".join(words)  
    # 写入TSV文件  
    qid_idx_writer.write(str(qid) + "\t" + yishen_fact_truncated + "\n")  
  
qid_idx_writer.close()    
print(cnt)

