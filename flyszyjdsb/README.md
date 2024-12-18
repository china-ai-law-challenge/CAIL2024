# CAIL 2024  
# 法律要素抽取+争论焦点生成

### 任务描述
本任务要求参赛者设计系统，从真实的法律调解多轮对话数据中正确提取对话涉及的**法律要素**和调解方和被告方之间的**争议焦点**信息。

**法律要素**指在该对话中涉及的，对确定法律关系性质和认定法律责任起到关键作用的因素。一段对话中往往会涉及多个法律要素。同时，涉及不同案由的调解对话会包含不同的法律要素。例如，对于与信用卡纠纷相关的调解，法律要素包含信用卡违约的账单金额、本金、和银行提出的调解金额等三类法律要素。

**争议焦点**指在该调解对话中涉及的，对话参与者具有不同主张，且对于案件定性和归责具有决定性意义的事项。例如，在信用卡纠纷相关的调解中争议焦点往往是被告方是否愿意接受调解人员给出的还款方案，调解是否达成合意，以及是否能解除微信账号冻结等其他事项。

本任务的原始数据为电话录音相应的转录文本（可能包含一定噪声）。参赛者需要设计系统和算法，对对话进行理解、对其中论辩关系进行逻辑分析、并最终生成符合要求的法律要素和争议焦点。组织者会基于参赛者生成的法律要素和争议焦点计算与来自专业人员的人工标注的语义相似程度，对参赛系统进行评价。

### 数据介绍
本任务所采用的输入数据为法院工作人员与被告人的电话调解录音相应的人工校对生成的文本。

初赛阶段，组织者为参赛者提供涉及信用卡纠纷案由的开发集（dev set）和测试集（test set）。

复赛阶段，组织者为参赛者提供涉及信用卡纠纷、房屋租赁合同纠纷、物业合同纠纷、服务合同纠纷、买卖合同纠纷五种案由的开发集（dev set）和测试集（test set）。

初赛开发集包含28条电话录音的转录文本(TXT文件)以及来自专业标注者的法律要素和争议焦点标注，供参赛者调试模型和系统。以下展示了一条数据对应的转录文本（部分）、法律要素标注和争议焦点标注。

初赛测试集中包含70条电话录音的转录文本(TXT文件)，参赛者需要提取测试集转录文本中涉及的法律要素和争议焦点，提交到赛事官网进行自动评价。

初赛开发集和测试集请在赛事官网下载。

**示例**

**转录文本（部分）**

```
……

角色A：好的，我这边告诉您一下，您目前这边是有一个兴业银行那边提供的一个分期的方案，就是说您要分期首期要还到，按25,500块钱进行一个分期，您总金额是28,000多，25,000一个分期。

角色B：我总共额度都没有那么高，哪里需要还那么多钱，我在想。

角色A：您现在连本带利的总金额是28,077.25。

角色B：现在我也不管他们怎么算的，反正我肯定是要等会儿，现在我还没确认，我这几天事情比较多，也没有去具体看。没事，我打开看一下。

……

角色A：嗯？

角色B：我看一下，看一下余额。这个固定额度是25,000，但是我不可能超过25,000呀。

角色A：您的本金是24,762.81，您这个28,000都是您的连本带息的，理解吗？因为您这个金额有几万，然后日期时间有几年，一直没有处理。

角色B：如果是说这样子，就是说他银行那个意思是什么样的？

角色A：您现在连本带息是28,000多，然后您在调解期限内能够处理的话，就可以给您按照25,500进行一个处理。然后您之前也说了，这个期限是处理不了，之前调解员也反馈了，然后跟兴业银行那边协商的一个结果，就是说可以给您一个分期的方案，但是您首期要还个9500，后面是可以给您慢慢的去还。就是说您还完首期之后，包括您任何的这边，只要您还完首期，兴业银行那边就可以撤案，包括您的微信解冻程序也可以开始运转，理解吗？

角色B：那总共要还多少钱呢？

角色A：总共就是按25,500进行还款。

角色B：那后面法律利息什么东西的吗？

角色A：没有，只要您按照25,500去还，然后到时候这边会进行给您进行一个司法确认，我们这边会由您以及申请方那边的一个诉讼代理人，以及端州区这边的一个主审法官，你们三方会进行一个司法确认签字，包括我们这边是要签订一个分期协议，只要您不违约，这个金额就不会再变的。但如果说您再次单方违约的话，后面肯定会重新算的好吧？
```

**输出**

**法律要素**

```
[
    “案由：信用卡纠纷”,
    “信用卡欠款账单金额28,077.25”,
    “信用卡欠款本金24,762.81”,
    “调解方案还款金额25,500”,
]
```

**争议焦点**

```
调解方案为按按25,500分期偿还，首期9500；被告方无法支付首期；调解双方未达成合意。
```

### 评价方式
对于法律要素生成任务，专家标注的标准答案为一个对话中所有涉及的多个法律要素组成的集合：G={g_1, g_2, …, g_n}。相应的，参赛者提交的答案应为一个包含所有涉及法律要素的集合: E={e_1, e_2,…,e_m}。两个集合中的元素g_i和e_i均为一个表示法律要素的短文本（如：“信用卡欠款账单金额28,077.25 “）。在初赛阶段，我们会使用大模型自动判断G和E元素是否在语义上是否相似，进而判断标准答案G中任意一条法律要素是否被参赛系统有效识别。我们使用F1指标作为评价指标，其计算方法如下：

P=#{G中被正确识别的法律要素}/m

R=#{G中被正确识别的法律要素}/n

F1=2*P*R/(P+R)

对于争议焦点生成任务，专家标注的标准答案为一个短文本，参赛者提交的答案也应为一个短文本。因此，我们将使用ROUGE(Recall-Oriented Understudy for Gisting Evaluation)进行评价。

### 基线系统
我们将使用通用大语言模型（如qwen、GLM等）基于prompt engineering构建两个子任务的基线系统, 可自选模型类别，API_KEY请自行填写。我们会构建两个基线：

1）使用大模型直接生成对话中涉及的法律要素和争议焦点
```
python baseline/baseline.py -d 数据集目录 -o 输出文件  
```

2）使用大模型首先判断对话涉及的案由，然后再在给定案由的情况下，生成法律要素和争议焦点。
```
python baseline/baseline_withcause.py -d 数据集目录 -o 输出文件  
```
以开发集为例：
```
python baseline/baseline.py -d valid -o baseline_valid.json
python baseline/baseline_withcause.py -d valid -o baseline_withcause_valid.json
```

### 比赛时间线
1. 初赛阶段：9月13日至10月26日
2. 复赛阶段：10月27日至11月15日
3. 封测阶段：11月16日至11月31日

### 评测方式
本任务的评测方式为提交结果进行自动评测。
参赛者提交一份测试集结果的JSON文件，文件格式如下：
1. id: TXT文件名
2. legal_elements: 法律要素
3. dispute_focus: 争论焦点
```
[
    {
        "id": "240828-13",
        "legal_elements": [
            "案由：信用卡纠纷",
            "信用卡欠款账单金额28,077.25",
            "信用卡欠款本金24,762.81",
            "调解方案还款金额25,500",
        ],
        "dispute_focus": "调解方案为按按25,500分期偿还，首期9500；被告方无法支付首期；调解双方未达成合意。"
    },
    ...
]
```

### 赛程安排
比赛分为初赛阶段、复赛阶段和封测阶段。初赛阶段为9月13日至10月26日，复赛阶段设置为10月27日至11月15日，封测阶段设置为11月16日至11月31日。

在初赛阶段性能超过基线算法的队伍可以进入复赛。在复赛阶段表现最好的10个队伍会被邀请进行封闭测试。

### 评测负责同学及联系方式
公培元 pygongnlp@gmail.com

马晟杰 msj@ruc.edu.cn

### 封测结果提交形式
对【裁判文书事实生成】评测而言，采用了“服务-请求”方案进行评估。参赛者需要在本地搭建推理服务，之后评估方会通过“访问服务”的方式获取推理结果。参赛者所提供的推理服务须符合以下接口规范要求。

* 参赛者需要提供一个API的URL或域名，以便我们能够访问服务。例如：https://api.example.com/model.
* HTTP方法： API应当支持POST请求方法。
* 请求参数： API应接受一个JSON请求体，示例输入如下：
```
{
    "text": "<案例内容字符串>"
}
```
* 响应格式： API应当返回一个JSON响应，示例输出如下：
```
{
    'legal_elements':[<要素列表>], 
    'dispute_focus': "<争议焦点字符串>"
}
```
* 调用选手提供的API的示例：
```
def call_api(case, api_url):
    response = None
    try:
        response = requests.post(api_url, json={"text": case}, timeout=30)
    except Exception as e:
        raise LLMCallError("Failed to call the model API") from e

    # Check the response status code
    if response.status_code != 200:
        raise LLMCallError(f"API returned status code {response.status_code}")
    return response.json()
```
### 注意事项：
* 封测阶段已开放，截止时间为11月30日晚24点。
* API封装和部署完成后请自行先测试是否可以通过requests.post获取结果；
* 请确保测试期间您的具备稳定性和可靠性，以便在测试时能够正常访问和使用；
* 每条样例requests.post time不得超过30s，超过30s记当前样例输出结果为空。
* 为保证比赛公平公正，第一次成功提交封测阶段的成绩将作为最终成绩； 
* 各位选手需要提交程序源代码及模型（如果有），压缩后发到邮箱 2023000140@ruc.edu.cn, 相同评估脚本将离线测试结果，如果结果和平台结果不一致，将与选手对接问题，发现作弊行为或不提交将取消成绩。
