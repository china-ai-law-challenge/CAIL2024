from tqdm import tqdm
import os
import json
import jieba


file_path = 'workplace/CAIL_Task4/data/sample_data/cleaned_cases_fact.json'
with open(file_path, 'r', encoding='utf-8') as file:
    documents = json.load(file)

file_path = 'workplace/CAIL_Task4/data/sample_data/new_queries_fact.json'
with open(file_path, 'r', encoding='utf-8') as file:
    queries = json.load(file)


# print(len(documents))

document_dict = []
for query in tqdm(documents):
    out_dict = {}
    out_dict["id"] = query["id"]
    yishen = query["yishen_fact"].replace('\n', '').replace('\t', '')
    ershen = query["ershen_fact"].replace('\n', '').replace('\t', '')
    content = yishen[:512] + ' ' +ershen[0:512]
    content = content.replace('\n', '').replace('\t', '')
    words = jieba.lcut(content, cut_all=False)
    cutted_psg = ' '.join(words)
    out_dict['contents'] = cutted_psg
    document_dict.append(out_dict)
    
with open('workplace/CAIL_Task4/data/documents/documents_tokenization.json', 'w', encoding='utf-8') as f:
    json.dump(document_dict, f, ensure_ascii=False, indent=4)
    

