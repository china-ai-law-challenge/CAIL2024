import argparse
import os
import json
from tqdm import tqdm
from zhipuai import ZhipuAI

from construct_json import try_parse_json_object

client = ZhipuAI(api_key="")  # 输入GLM4 API Key


def extract_single_json(laws):
    try:
        _, laws = try_parse_json_object(laws)
        return laws
    except Exception as e:
        lines = laws.split("\n\n")
        for line in lines:
            try:
                _, laws = try_parse_json_object(line)
                return laws
            except json.JSONDecodeError:
                continue
    return None


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
    parser.add_argument("-d", "--dataset_path", type=str, default="dataset")
    parser.add_argument("-p", "--prompt_path", type=str, default="prompts")
    parser.add_argument("-m", "--model_name", type=str, default="GLM-4")
    args = parser.parse_args()

    for file_path in tqdm(os.listdir(args.dataset_path)[85:]):
        file_path = os.path.join(args.dataset_path, file_path)
        conv = open(file_path, 'r', encoding='utf8').read().strip('\n')

        prompt_path = os.path.join(args.prompt_path, 'extract_laws.txt')
        prompt = open(prompt_path, 'r', encoding='utf8').read()
        prompt = prompt.replace('[conv]', conv)
        laws = generate(args.model_name, prompt)
        laws = extract_single_json(laws)
        print(laws)
