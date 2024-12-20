import os
from fastapi import HTTPException
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from tempfile import NamedTemporaryFile

# 初始化模型
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, "..", "models", "SenseVoiceSmall")
import subprocess

def get_gpu_info():
    try:
        # 运行 nvidia-smi 命令并捕获输出
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'])
        # 解码输出为字符串并去除末尾的换行符
        gpu_indices = output.decode('utf-8').strip().split('\n')
        return gpu_indices
    except subprocess.CalledProcessError:
        # 如果 nvidia-smi 命令执行失败，返回一个空列表
        return []

def initialize_model():
    # 获取显卡编号列表
    gpu_indices = get_gpu_info()
    
    # 如果有可用的显卡，选择第一个显卡
    if gpu_indices:
        device = f"cuda:{gpu_indices[0]}"
    else:
        # 如果没有可用的显卡，使用CPU
        device = "cpu"
    model = AutoModel(
        model=model_dir,
        trust_remote_code=True,
        vad_kwargs={"max_single_segment_time": 30000},
        device=device,
        hub="hf",
        disable_update=True  # 禁用更新检查
    )
    return model

model = initialize_model()
emo_set = {"😊", "😔", "😡", "😰", "🤢", "😮"}
event_set = {"🎼", "👏", "😀", "😭", "🤧", "😷"}

def transcribe_audio(audio_file: str) -> str:
    """
    转写音频文件
    """
    try:
        res = model.generate(
            input=audio_file,
            cache={},
            language="auto",  # 自动检测语言
            use_itn=True,  # 输出结果中是否包含标点与逆文本正则化
            batch_size_s=60,  # 动态批处理音频时长（单位秒）
            merge_vad=True,  # 合并 VAD 分段
            merge_length_s=15,  # 合并后的音频片段长度
        )
        text = rich_transcription_postprocess(res[0]["text"])
        # 手动去除情绪符号
        text = ''.join([char for char in text if char not in emo_set and char not in event_set])
        return text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"语音转写失败: {str(e)}")

from fastapi import UploadFile
from pydub.utils import mediainfo
async def save_audio_file(file: UploadFile) -> str:
    """
    保存上传的音频文件并返回文件路径，同时检查文件的格式是否合法
    """
    filename = file.filename
    save_dir = os.path.join(os.path.dirname(__file__), "tmp")
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, filename)
    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
    # 检查文件的 MIME 类型或格式
    try:
        info = mediainfo(file_path)  # 获取文件的详细信息
        if info.get("codec_type") != "audio":
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid audio file. Please upload a valid audio file.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing audio file: {str(e)}")
    return file_path