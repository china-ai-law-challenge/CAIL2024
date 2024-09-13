import json
import re

file_path = 'workplace/CAIL_Task4/data/sample_data/cleaned_cases.json'
with open(file_path, 'r', encoding='utf-8') as file:
    cases = json.load(file)

file_path = 'workplace/CAIL_Task4/data/bm25_retrieve_results.json'
with open(file_path, 'r', encoding='utf-8') as file:
    bm25_reuslts = json.load(file)


def get_reason(content):
    pattern = r'本院认为.*?(?=本院认为|$)'

    # 使用 re.findall 来找到所有匹配的内容
    matches = re.findall(pattern, case["ershen_content"], re.DOTALL)

    # 检查是否找到匹配项
    if matches:
        # 如果有多个匹配项，我们取最后一个
        match_str = matches[-1]
    else:
        match_str = "none"

    return match_str

reasons_dict = []
for i in range(len(bm25_reuslts)):
    out_dict = {}
    top1_case_id = bm25_reuslts[i]["document_ids"][0]
    out_dict['qid'] = str(bm25_reuslts[i]["id"])
    for case in cases:
        if str(case["id"]) == top1_case_id:
            reason = get_reason(case["ershen_content"])
            out_dict['reason'] = reason
    reasons_dict.append(out_dict)

with open('workplace/CAIL_Task4/data/reason_prediction.json', 'w', encoding='utf-8') as f:
    json.dump(reasons_dict, f, ensure_ascii=False, indent=4)

