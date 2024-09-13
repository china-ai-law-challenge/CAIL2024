import json
from pathlib import Path
import os
from transformers import AutoModel,AutoTokenizer
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
# from model.modeling_chatglm import ChatGLMForConditionalGeneration
# from model.tokenization_chatglm import ChatGLMTokenizer

def count_tokenizer_avg():
    file_path = 'denoted_data_50_99.json'
    field_name = 'fact'
    
    # model = ChatGLMForConditionalGeneration.from_pretrained("LexiLaw/model", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("LexiLaw/model", trust_remote_code=True)
    total_tokens = 0
    count = 0
    with open(file_path, 'r') as f:
        data = json.load(f)
        total_tokens = 0
        count = 0
        for item in data:
            if field_name in item:
                text = item[field_name]
                tokens = tokenizer.encode(text, return_tensors='pt')[0]
                total_tokens += len(tokens)
                count += 1
        if count > 0:
            return total_tokens / count
        else:
            return 0

    


def count_field_length_avg():
    file_path = 'denoted_data_50_99.json'
    field_name = 'fact'
    lengths = []
    with open(file_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
        for data in data_list:
            if field_name in data:
                lengths.append(len(data[field_name]))
    if lengths:
        return sum(lengths) / len(lengths)
    else:
        return 0

def writeSth(folder_path,output_path):
    writeField(folder_path,output_path,"reasoning")
    writeField(folder_path,output_path,"judgement")
    writeField(folder_path,output_path,"cause")
    writeField(folder_path,output_path,"ethics_or_jurisprudence")

def writeField(folder_pathstr,output_path,field):
    folder_path=Path(folder_pathstr)
    instruction=""
    answer=""
    # if(field == "reasoning"):
    #     instruction="假设你是法律领域专家。"+"请根据法律案件的事实部分给出其司法判决说理内容,同时给出其涉及的伦理或法理。"  +"要求相关事实在事实文本中所在的具体句子位置与该事实所对应的伦理或法理相对应。每句话以句号视为一句话，每当出现一个句号，我们视为该句话结束。其中第一句话用0表示，第二句话用1表示，以此类推"+"下面是一个示例："  +"事实部分：本院查明如下事实：原告青岛三金机电有限公司与被告中建新疆建工（集团）有限公司买卖合同纠纷一案2019年至2022年期间，原被告间签订多份物资采购合同，约定原告为被告施工的项目供应电线电缆、沟槽管件等建筑材料和设备，原告已经按约履行供货义务，被告仅支付部分货款，截止原告起诉，被告尚欠货款695137.05元未付。为维护原告合法权益，特诉至法院。被告辩称，认可原告起诉欠款金额，但被告无力支付。当事人围绕诉讼请求依法提交了证据，本院组织当事人进行了证据交换和质证。根据举证、质证及认定证据。原被告双方分别就连云港灌云妇幼保健院项目、诸城恐龙探索王国博物馆项目、淄博市CBD项目商业地块二期项目、哈尔滨工程大学青岛创新发展基地一期项目、淄博CBD项目商业地块四期（MALL）项目、山钢.泓明府项目签订《物资采购与供应合同》共计十二份，约定原告为被告施工的上述项目供应电线电缆、沟槽管件等建筑材料和设备，合同签订后，原告按约供货，经双方结算以上十二份合同总供货金额为2075430.59元，经原被告双方当庭确认，被告已经付款1341000元，对于剩余的734430.59元，原告现仅要求被告支付695137.05元，其余的39293.54元系原被告双方结算扣减费用，原告当庭陈述不再主张。"+"判决说理部分：本院认为，依法成立的合同受法律保护。经查本案中原被告双方签订的《物资采购与供应合同》系双方真实意思表示，且不违反法律法规强制性规定，合法有效，双方均应依约履行各自的义务。经查，原告已按照合同约定履行供货义务，经双方结算当庭确认总供货金额为2075430.59元，被告已经付款1341000元，剩余的其中39293.54元原告当庭陈述不再主张，故对于原告要求被告支付货款695137.05元的主张，事实清楚，证据充分，本院予以支持。综上，依据《中华人民共和国民法典》第四百六十五条、第五百零二条、第五百零九条、第五百七十九条之规定，判决如下：被告中建新疆建工（集团）有限公司于本判决生效后10日内支付原告青岛三金机电有限公司货款695137.05元。如果未按本判决指定的期间履行给付金钱的义务，应当按照《中华人民共和国民事诉讼法》第二百六十条之规定，加倍支付迟延履行期间的债务利息。案件受理费减半收取5376元，由被告负担，被告在履行上述付款义务时将应承担的案件受理费一并给付原告。"+"伦理或法理:0:  依法成立的合同受法律保护,1: 合同合法有效，双方应按照所签订的合同履行各自的义务(0,1分别对应判决说理部分的第一句话，第二句话)"
    # elif(field == "judgement"):
    #     instruction="假设你是法律领域专家。"+"请根据法律案件的事实部分给出其司法判决结果。下面是一个示例："    +"事实部分：本院查明如下事实：原告青岛三金机电有限公司与被告中建新疆建工（集团）有限公司买卖合同纠纷一案2019年至2022年期间，原被告间签订多份物资采购合同，约定原告为被告施工的项目供应电线电缆、沟槽管件等建筑材料和设备，原告已经按约履行供货义务，被告仅支付部分货款，截止原告起诉，被告尚欠货款695137.05元未付。为维护原告合法权益，特诉至法院。被告辩称，认可原告起诉欠款金额，但被告无力支付。当事人围绕诉讼请求依法提交了证据，本院组织当事人进行了证据交换和质证。根据举证、质证及认定证据。原被告双方分别就连云港灌云妇幼保健院项目、诸城恐龙探索王国博物馆项目、淄博市CBD项目商业地块二期项目、哈尔滨工程大学青岛创新发展基地一期项目、淄博CBD项目商业地块四期（MALL）项目、山钢.泓明府项目签订《物资采购与供应合同》共计十二份，约定原告为被告施工的上述项目供应电线电缆、沟槽管件等建筑材料和设备，合同签订后，原告按约供货，经双方结算以上十二份合同总供货金额为2075430.59元，经原被告双方当庭确认，被告已经付款1341000元，对于剩余的734430.59元，原告现仅要求被告支付695137.05元，其余的39293.54元系原被告双方结算扣减费用，原告当庭陈述不再主张。"+"判决结果部分：被告中建新疆建工（集团）有限公司于本判决生效后10日内支付原告青岛三金机电有限公司货款695137.05元。如果未按本判决指定的期间履行给付金钱的义务，应当按照《中华人民共和国民事诉讼法》第二百六十条之规定，加倍支付迟延履行期间的债务利息。案件受理费减半收取5376元，由被告负担，被告在履行上述付款义务时将应承担的案件受理费一并给付原告"
    # elif(field == "cause"):
    #     instruction="假设你是法律领域专家。"+"请根据法律案件的事实部分给出其司法案由。下面是一个示例："+"事实部分：本院查明如下事实：原告青岛三金机电有限公司与被告中建新疆建工（集团）有限公司买卖合同纠纷一案2019年至2022年期间，原被告间签订多份物资采购合同，约定原告为被告施工的项目供应电线电缆、沟槽管件等建筑材料和设备，原告已经按约履行供货义务，被告仅支付部分货款，截止原告起诉，被告尚欠货款695137.05元未付。为维护原告合法权益，特诉至法院。被告辩称，认可原告起诉欠款金额，但被告无力支付。当事人围绕诉讼请求依法提交了证据，本院组织当事人进行了证据交换和质证。根据举证、质证及认定证据。原被告双方分别就连云港灌云妇幼保健院项目、诸城恐龙探索王国博物馆项目、淄博市CBD项目商业地块二期项目、哈尔滨工程大学青岛创新发展基地一期项目、淄博CBD项目商业地块四期（MALL）项目、山钢.泓明府项目签订《物资采购与供应合同》共计十二份，约定原告为被告施工的上述项目供应电线电缆、沟槽管件等建筑材料和设备，合同签订后，原告按约供货，经双方结算以上十二份合同总供货金额为2075430.59元，经原被告双方当庭确认，被告已经付款1341000元，对于剩余的734430.59元，原告现仅要求被告支付695137.05元，其余的39293.54元系原被告双方结算扣减费用，原告当庭陈述不再主张。"+"案由部分：买卖合同纠纷"
    

    if(field == "reasoning"):
        instruction="假设你是法律领域专家。"+"请根据法律案件的事实部分给出其司法判决说理内容。"
    elif(field == "judgement"):
        instruction="假设你是法律领域专家。"+"请根据法律案件的事实部分给出其司法判决结果。"
    elif(field == "cause"):
        instruction="假设你是法律领域专家。"+"请根据法律案件的事实部分给出其司法案由。"
    elif(field == "ethics_or_jurisprudence"):
        instruction = "假设你是法律领域专家,请根据法律案件的司法判决说理给出其涉及的伦理或法理，要求相关事实在事实文本中所在的具体句子位置与该事实所对应的伦理或法理相对应。每句话以句号视为一句话，每当出现一个句号，我们视为该句话结束。其中第一句话用0表示，第二句话用1表示，以此类推"

    with open(output_path, 'a',encoding='utf-8') as output_file:
        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                file_path = os.path.join(folder_path,filename)
                # print(filename)
                # print(field)
                with open(file_path,'r',encoding='utf-8') as f:
                    data_list = json.load(f)
                    # flag = True
                    for item in data_list:
                        factGet = item.get('fact')
                        fact =  f"事实部分：{factGet}"
                        if(field=="judgement"):
                            judgement=item.get('judgement')
                            answer = f"判决结果部分：{judgement}"
                            new_item ={"instruction": instruction, "input":fact+"。 判决结果：", "answer": answer}
                        elif(field=="cause"):
                            cause=item.get("cause")
                            answer = f"案由部分：{cause}"
                            new_item ={"instruction": instruction, "input":fact+"。 案件案由：", "answer": answer}
                        elif(field=="reasoning"):
                            reasoning=item.get('reasoning')
                            # etics=item.get('ethics_or_jurisprudence')
                            answer = f"判决说理部分：{reasoning}"
                            new_item ={"instruction": instruction, "input":fact+"。 判决说理：", "answer": answer}
                        elif(field=="ethics_or_jurisprudence"):
                            reasoning=item.get('reasoning')
                            fact = f"判决说理部分：{reasoning}"
                            ethics_or_jurisprudence=item.get('ethics_or_jurisprudence')
                            answer = f"伦理或法理部分：{ethics_or_jurisprudence}"
                            new_item ={"instruction": instruction, "input":fact+"。 伦理法理：", "answer": answer}
                        # if(flag):
                        #     flag=False
                        #     print(answer)
                        output_file.write(json.dumps(new_item,ensure_ascii=False) + '\n')
def main():
    folder_path = 'dataDeal/raw'
    output_path='train.json'
    writeSth(folder_path,output_path)
    # print(count_tokenizer_avg())


if __name__ == '__main__':
    main()
