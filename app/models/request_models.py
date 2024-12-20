from pydantic import BaseModel

# 自我介绍请求体
class IntroductionRequest(BaseModel):
    introduction: str

# 面试问答请求体
class InterviewAnswerRequest(BaseModel):
    question: str
    answer: str
