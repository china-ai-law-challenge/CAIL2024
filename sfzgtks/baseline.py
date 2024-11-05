# encoding=utf-8
import json
import os
import dashscope
from dashscope import Generation

prompt_template = """
###任务：
你是一个司法考试助手，将对司法考试问题进行回答。

###要求：
1.回复使用法言法语和专业术语；
2.回复务必保证语言简洁、逻辑清晰。

###问题：
{question}
"""

def ask_llm(prompt):
    """
    call your qwen_api and return response
    example: call qwen api from Bailian (https://help.aliyun.com/zh/model-studio/getting-started/what-is-model-studio?spm=a2c4g.11174283.0.i3)
    """
    response = Generation.call(
        api_key=os.getenv("DASHSCOPE_API_KEY", "add your api key here"),
        model="qwen2.5-7b-instruct",
        messages=[{'role': 'user', 'content': prompt}],
        result_format='message'
    )
    if response.status_code == 200:
        return response.output.choices[0].message.content
    else:
        print(f"HTTP返回码：{response.status_code}")
        print(f"错误码：{response.code}")
        print(f"错误信息：{response.message}")
        print("请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code")
        return None


def get_res(input_file, output_file):
    lines = open(input_file).readlines()
    with open(output_file, 'w') as f_out:
        for line in lines:
            data = json.loads(line)
            id = data['id']
            big_ques = data['big_ques']
            small_ques = data['small_ques']
            prompt = prompt_template.format(question=big_ques + "\n" + small_ques)
            prompt = str(prompt).strip()
            user_answer = ask_llm(prompt=prompt)
            j = {"id": id, "user_answer": user_answer}
            f_out.write(json.dumps(j, ensure_ascii=False) + "\n")
            f_out.flush()


if __name__ == '__main__':
    get_res('test_data.json', 'prediction.json')

