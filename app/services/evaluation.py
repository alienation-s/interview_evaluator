from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
from sparkai.core.messages import ChatMessage
import time
import yaml
import os

# 加载配置
current_dir = os.path.dirname(os.path.abspath(__file__))
config_file_path = os.path.join(current_dir, '../config.yaml')
# 加载 YAML 配置文件
def load_config(config_file=config_file_path):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


config = load_config()

# 从配置中提取相应的参数
SPARKAI_URL = config['sparkai']['url']
SPARKAI_APP_ID = config['sparkai']['app_id']
SPARKAI_API_SECRET = config['sparkai']['api_secret']
SPARKAI_API_KEY = config['sparkai']['api_key']
SPARKAI_DOMAIN = config['sparkai']['domain']

spark = ChatSparkLLM(
    spark_api_url=SPARKAI_URL,
    spark_app_id=SPARKAI_APP_ID,
    spark_api_key=SPARKAI_API_KEY,
    spark_api_secret=SPARKAI_API_SECRET,
    spark_llm_domain=SPARKAI_DOMAIN,
    streaming=False,
    temperature=0.01
)

def get_model_response(messages, handler):
    start_time = time.time()
    a = spark.generate([messages], callbacks=[handler])
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"推理及响应的时间共： {execution_time:.4f} 秒")
    return a.generations[0][0].message.content.strip()

def evaluate_self_intro(introduction: str):
    prompt = """
    你是一名资深的面试官。你将收到一个面试者的自我介绍。
    请你对面试者的自我介绍进行评估打分（0-100分），并给出自我介绍的修改建议。
    你将基于以下方面进行综合打分与反馈：项目/实习经历0-25分、技能掌握0-25分、教育背景0-25分、整体表达0-25分。请确保您的分析既客观又具体，对每次的评估打分标准一致，给出每个方面的分数，最后给出总分（满分100分），并针对该自我介绍提出修改或强化建议。请严格按照输出格式输出，不要输出无关的内容和你的思考过程。

    以下是评分标准：
    1. 项目/实习经历: 评估面试者在项目中、实习中的角色重要性、所负责任务的复杂性、使用的技能以及取得的成果。
    2. 技能掌握: 根据面试者列出的技能与应聘岗位的相关性和深度进行评价。
    3. 教育背景: 考虑面试者的毕业院校声誉、其所学专业与应聘职位的相关度、参与的学生工作经历、在校期间获得的比赛奖项和奖学金情况。
    4. 整体表达: 评估自我介绍的逻辑性、清晰度以及是否能够突出面试者的优势。
    5. 禁止输出无关内容和你的思考过程。

    请严格按照以下输出格式回答，改进建议详细一些即可:
    总分: 0-100分
    项目/实习经历评分: 0-25分
    技能掌握评分: 0-25分
    教育背景评分: 0-25分
    整体表达评分: 0-25分
    改进建议:

    请评估以下自我介绍：
    """ + introduction
    messages = [ChatMessage(role="user", content=prompt)]
    handler = ChunkPrintHandler()
    response = get_model_response(messages, handler)
    return response

# 处理面试问答并返回评分
def evaluate_answer(question, answer):
    prompt = """
    你是一个计算机领域专家面试官。你将收到一个问题、参考答案和面试者回答。
你的任务是通过将面试者回答与参考答案和你的现有知识进行比较，评价面试者回答好坏的分数。
请从四个方面评价面试者的回答，总分数score为0-100分，总分其中包括准确性0-50分、覆盖全面性0-20分、细节全面性0-20分、内容相关性0-10分，分数越高表示整体表现越好。首先你要给出四个方面的单独评分，最后你需要对四个方面评分进行求和，给出总分score。
请输出面试者在这四个方面的具体得分，最后返回求和的最终结果。
请你在评估的过程中避免任何潜在的偏见，并确保回答出现的顺序不会影响你的判断。
四个方面评价标准和步骤如下：
1.详细阅读提交的问题。
2.思考这个问题，并阅读给定的参考答案，确保理解了问题的内容以及参考答案的含义。
3.阅读面试者的回答。
4.比较面试者的回答与参考答案和你的知识，理解面试者回答中的信息和参考答案的相似程度以及不同之处。
5.评价面试者的回答，输出面试者在这四个方面的具体得分，最后返回求和后的最终分数score。四个方面评分参考准则：
一：覆盖全面性(0-20分)：评估回答是否全面覆盖了问题所涉及的所有关键领域、方面或要点，没有显著遗漏。
覆盖全面性评分细则:
20分:回答极其全面，涵盖了所有关键领域、方面或要点，无一遗漏。
16分左右:回答非常全面，覆盖了绝大多数关键领域、方面或要点，仅有极个别非核心点未提及。
12分左右:回答较为全面，覆盖了主要问题的大部分关键领域、方面或要点，但有一些较重要的点被遗漏。
8分左右:回答部分全面，覆盖了问题的一部分关键领域、方面或要点，但遗了较多重要内容。
4分及以下:回答极不全面，几乎未覆盖问题的关键领域、方面或要点。

二：细节全面性(0-20分)：评估回答在覆盖的各个关键领域、方面或要点上，是否提供了足够的细节支持，使内容更加丰满和具体。
细节全面性评分细则:
20分:每个关键领域、方面或要点都有详尽的阐述和深入的分析，细节丰富。
16分左右：大部分关键领域、方面或要点有较为详细的解释，但个别地方略显略。
12分左右:有一定数量的关键领域、方面或要点有细节支持，但整体细节不够充分。
8分左右:少数关键领域、方面或要点有细节描述，但大多数内容较为笼统。
4分及以下:几乎未提供任何细节支持，内容空洞。

三：准确性(0-50分)：评估回答中提供的信息、数据、事实、理论或结论的准确程度。考察回答是否避免了误导性信息、错误陈述或夸大其词。
准确性评分细则：
50分:回答中的所有信息、数据、事实、理论或结论均准确无误，无任何误导性内容，且信息来源可靠。
40分左右:回答中的信息、数据、事实、理论或结论大部分准确，仅有一处或几处非核心细节存在微小的不准确，不影响整体理解。
30分左右:回答中大部分内容准确，但有一处或几处较为重要的信息存在不准确或模糊的情况，可能对理解产生一定影响。
20分左右:回答准确性一般，存在几处明显的错误或误导性信息，但整体方向仍然正确
10分及以下:回答中存在多处显著错误或误导性信息，严重影响对问题的正确理解或评估。

四：内容相关性(0-10分)：评估回答内容与问题要求的相关程度。
内容相关性评分细则:
10分:回答与问题高度相关，直接且准确地回应了问题的核心要点，没有偏离主题。
8分左右:回答与问题较为相关，主要内容与问题紧密相关，但可能有一处或几处非核心部分稍有偏离。
6分左右:回答与问题有一定的相关性，但整体而言，部分内容与问题核心要点的关联性不强。4分左右:回答与问题相关性较低，大部分内容未直接关联到问题的核心要点。
2分及以下:回答与问题几乎无关，未能回应问题的核心要点。

6.请严格根据我的上述评分标准和评分细则评分。
7.请严格按照我的示例的相同格式输出。
8.禁止输出无关内容和你的思考过程。
9.不要受我的示例的结果影响。
10.请严格按照我的示例的相同格式输出。
11.打分严格一些，在面试者回答的模糊时，给分低一些。并给出中肯的建议。
请参考我给出的示例如下：
总分(100分): 0-100分
覆盖全面性(20分): 0-20分
细节全面性(20分): 0-20分
内容准确性(50分): 0-50分
内容相关性(10分): 0-10分
改进建议：

下面是真实的问答，请根据示例和四个方面评价标准来回答：
    问题：{question}
    面试者回答：{answer}
    """.format(question=question, answer=answer)
    
    messages = [ChatMessage(role="user", content=prompt)]
    handler = ChunkPrintHandler()
    response = get_model_response(messages, handler)
    return response
