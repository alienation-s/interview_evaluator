import os
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from app.services.transcription import transcribe_audio, save_audio_file
from app.services.evaluation import evaluate_self_intro, evaluate_answer
from app.models.request_models import IntroductionRequest, InterviewAnswerRequest
from pydantic import BaseModel
from app.utils.audio_util import is_silent
from pydub import AudioSegment

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化 FastAPI 应用
app = FastAPI(debug=True)

# Pydantic 模型定义
class SuccessResponse(BaseModel):
    status: str
    message: str
    data: dict

class ErrorResponse(BaseModel):
    status: str
    message: str


# 音频转录接口
@app.post("/transcribe/", response_model=SuccessResponse)
async def transcribe_file(file: UploadFile = File(...)):
    logger.debug(f"Received file: {file.filename}")
    try:
        # 保存上传的音频文件
        audio_path = await save_audio_file(file)

        # 加载音频文件并检测是否静音
        audio = AudioSegment.from_file(audio_path)
        logger.debug(f"Audio length: {len(audio)} ms")

        # 如果音频是静音的，抛出 HTTPException
        if is_silent(audio):
            logger.debug("Audio is silent.")
            if os.path.exists(audio_path):
                os.remove(audio_path)
            raise HTTPException(
                status_code=400,
                detail="Audio is silent or blank. Please upload a valid audio file."
            )

        # 转写音频文件
        transcript = transcribe_audio(audio_path)
        logger.debug(f"Transcription: {transcript}")

        # 删除临时文件
        if os.path.exists(audio_path):
            os.remove(audio_path)

        return SuccessResponse(
            status="success", 
            message="Audio transcription completed successfully.",
            data={"text": transcript}
        )

    except HTTPException as e:
        # FastAPI 会自动返回 HTTPException 处理的响应
        logger.error(f"HTTPException occurred: {e.detail}")
        raise e  # 不需要返回 ErrorResponse，FastAPI 会处理 HTTPException

    except Exception as e:
        # 捕获其他异常并返回500错误
        logger.error(f"Unexpected error during transcription: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail="Unexpected error during transcription."
        )


# 自我介绍评估接口
@app.post("/evaluate/self-introduction/", response_model=SuccessResponse)
async def evaluate_self_introduction(data: IntroductionRequest):
    """
    评价面试自我介绍
    """
    try:
        # 清理自我介绍文本
        cleaned_introduction = " ".join(data.introduction.strip().split())
        logger.info(f"Received self-introduction for evaluation: {cleaned_introduction}")

        # 验证自我介绍长度
        if len(cleaned_introduction) < 10:
            raise HTTPException(status_code=400, detail="Introduction is too short, must be at least 10 characters.")
        
        # 验证是否为空或仅由空格组成
        if not cleaned_introduction:
            raise HTTPException(
                status_code=400, 
                detail="Introduction text cannot be empty or only whitespace. Please provide a valid self-introduction."
            )

        # 调用评估函数
        result = evaluate_self_intro(cleaned_introduction)
        return SuccessResponse(
            status="success", 
            message="Self-introduction evaluation successful.",
            data={"text": result}
        )

    except HTTPException as e:
        logger.error(f"HTTPException occurred: {e.detail}")
        raise e

    except Exception as e:
        logger.error(f"Unexpected error during evaluation: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail="Unexpected error during evaluation."
        )


# 面试问答评估接口
@app.post("/evaluate/interview-answer/", response_model=SuccessResponse)
async def evaluate_interview_answer(data: InterviewAnswerRequest):
    """
    评价面试问答
    """
    try:
        logger.info(f"Received interview answer evaluation request: {data}")
        cleaned_question = " ".join(data.question.strip().split())
        cleaned_answer = " ".join(data.answer.strip().split())
        # 验证数据是否为空
        if not cleaned_question or not cleaned_answer:
            raise HTTPException(status_code=400, detail="Question or answer cannot be empty.")

        result = evaluate_answer(cleaned_question, cleaned_answer)
        return SuccessResponse(
            status="success", 
            message="Interview answer evaluation successful.",
            data={"text": result}
        )

    except HTTPException as e:
        logger.error(f"HTTPException occurred: {e.detail}")
        raise e

    except Exception as e:
        logger.error(f"Unexpected error during evaluation: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail="Unexpected error during evaluation."
        )
