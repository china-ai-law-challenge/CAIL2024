import argparse
import os
import json
from tqdm import tqdm
from zhipuai import ZhipuAI

client = ZhipuAI(api_key="") # 请自行填写

cause_list = ["信用卡纠纷", "房产租赁合同纠纷", "买卖合同纠纷", "服务合同纠纷", "物业合同纠纷"]


def generate(model_name, prompt):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0
    )
    return response.choices[0].message.content


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_path", type=str, required=True)
    parser.add_argument("-o", "--output_path", type=str, required=True)
    parser.add_argument("-p", "--prompt_path", type=str, default="prompts")
    parser.add_argument("-m", "--model_name", type=str, default="GLM-4-FLASH")
    args = parser.parse_args()

    results = []
    for file_path in tqdm(os.listdir(args.dataset_path)):
        file_name = file_path.strip(".txt")
        file_path = os.path.join(args.dataset_path, file_path)
        conv = open(file_path, 'r', encoding='utf8').read().strip('\n')

        cause_prompt_path = os.path.join(args.prompt_path, 'generate_cause.txt')
        cause_prompt = open(cause_prompt_path, 'r', encoding='utf8').read()
        cause_prompt = cause_prompt.replace('[conv]', conv)
        cause_text = generate(args.model_name, cause_prompt)
        matched_causes = [cause for cause in cause_list if cause in cause_text]
        cause = matched_causes[0]

        prompt_path = os.path.join(args.prompt_path, 'extract_laws_withcause.txt')
        prompt = open(prompt_path, 'r', encoding='utf8').read()
        prompt = prompt.replace('[conv]', conv)
        prompt = prompt.replace('[cause]', cause)
        laws = generate(args.model_name, prompt)

        prompt_path = os.path.join(args.prompt_path, 'generate_points_withcause.txt')
        prompt = open(prompt_path, 'r', encoding='utf8').read()
        prompt = prompt.replace('[conv]', conv)
        prompt = prompt.replace('[cause]', cause)
        point = generate(args.model_name, prompt)
        point = point.lstrip("争议焦点：").strip("\n")
        if len(point.split("\n")) != 1 and "争议焦点" in point.split("\n")[0]:
            point = "".join(point.split("\n")[1:])

        results.append({
            "id": file_name,
            "legal_elements": laws.split("\n"),
            "dispute_focus": point
        })

    json.dump(results, open(args.output_path, "w", encoding="utf8"), ensure_ascii=False, indent=4)
