from tqdm import *
import json
    
    
class GLM4_API():
    """
    Add your chatGLM4_API
    """

glm4 = GLM4_API()

def conversation2str(con):
    """
    Transfer the conversation to string
    """
    res = ""
    for c in con:
        res += f"{c['role']}: {c['content']}\n"
    return res

def get_res(outputpath):
    """
    Get the baseline res
    """
    f_out = open(outputpath, 'w')
    test_data = json.load(open('./test_data.json', 'r'))
    prompt_template =  open("./prompt.txt", encoding='utf-8').read().strip()
    for data in tqdm(test_data):
        prompt = prompt_template
        prompt = prompt.replace('{{question_text}}', conversation2str(data['conversation']))
        r = glm4.chat(prompt)
        l = {"id": data['id'], "response": r}
        f_out.write(json.dumps(l, ensure_ascii=False) + '\n')

get_res('./prediction.json')