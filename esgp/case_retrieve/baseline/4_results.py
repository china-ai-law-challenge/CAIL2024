import csv
import json

# TSV文件路径  
tsv_file_path = 'workplace/CAIL_Task4/data/sample_data/bm25_output.tsv'  

dict = []
# 打开TSV文件并读取  
with open(tsv_file_path, mode='r', encoding='utf-8', newline='') as tsv_file:  
    tsv_reader = csv.reader(tsv_file, delimiter='\t')  
      
    # 遍历文件中的每一行  
    i = 0
    for row in tsv_reader: 
        if i % 5 ==0:
            ids = []
        # row是一个列表，包含当前行的所有字段  
        # 假设第一列是ID，第二列是处理后的查询文本  
        print(row[0])
        ids.append(row[0].split()[2])  
        
        if len(ids) == 5:
            out_dict = {
                "id": row[0].split()[0],
                "document_ids": ids
            }
            dict.append(out_dict)
            
        i+=1
     
with open('workplace/CAIL_Task4/data/bm25_retrieve_results.json', 'w', encoding='utf-8') as f:
    json.dump(dict, f, ensure_ascii=False, indent=4)
       
