from tqdm import *
import json
    
    
class GLM4_API():
    """
    Add your chatGLM4_API
    """

glm4 = GLM4_API()

def get_res(output_path):
    f_out = open(output_path, 'w')
    test_data = json.load(open('./test_data.json', 'r'))
    prompt_template =  open("./prompt_event.txt", encoding='utf-8').read().strip()
    prompt_template2 = open("./prompt_fact.txt", encoding='utf-8').read().strip()
    for data in tqdm(test_data):
        prompt = prompt_template
        prompt = prompt.replace('{{prosecution}}', data['prosecution'])
        prompt = prompt.replace('{{defense}}', data['defense'])
        event = {}
        event_str = ""
        for evidence_key in data['evidence']:
            evidence = data['evidence'][evidence_key]
            prompt_sub = prompt.replace('{{evidence}}', evidence)
            r = glm4.chat(prompt_sub)
            event[evidence_key] = r
            event_str += r 
            event_str += '\n'

        # 根据event生成fact
        prompt_fact = prompt_template2
        prompt_fact = prompt_fact.replace('{{event}}', event_str)
        r = glm4.chat(prompt_fact)

        l = {"id": data['id'], "fact": r, "event": event}

        f_out.write(json.dumps(l, ensure_ascii=False) + '\n')

get_res('./prediction.json')