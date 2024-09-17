from zhipuai import ZhipuAI
from tqdm import tqdm
import argparse
import os

client = ZhipuAI(api_key="")  # 输入GLM4 API Key


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

    for file_path in tqdm(os.listdir(args.dataset_path)):
        file_path = os.path.join(args.dataset_path, file_path)
        conv = open(file_path, 'r', encoding='utf8').read().strip('\n')

        prompt_path = os.path.join(args.prompt_path, 'generate_points.txt')
        prompt = open(prompt_path, 'r', encoding='utf8').read()
        prompt = prompt.replace('[conv]', conv)
        point = generate(args.model_name, prompt)
        point = point.lstrip("争议焦点：").strip("\n")
        if len(point.split("\n")) != 1 and "争议焦点" in point.split("\n")[0]:
            point = "".join(point.split("\n")[1:])
        print(point)
