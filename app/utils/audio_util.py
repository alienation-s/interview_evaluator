from pydub import AudioSegment
import numpy as np

# 生成静音
def generate_silence(duration=0):
    silence = AudioSegment.silent(duration=duration)
    return silence

# 保存音频为mp3文件
def save_as_mp3(audio, output_path):
    audio.export(output_path, format="mp3")

import numpy as np
from pydub import AudioSegment

def is_silent(audio: AudioSegment, silence_threshold=-40.0, sample_threshold=1e-3) -> bool:
    # 检查音频的时长是否为0（即没有音频数据）
    if len(audio) == 0:
        return True 
    # 获取音频的响度（dBFS）
    dBFS = audio.dBFS
    print(f"Audio dBFS: {dBFS}")  # 调试输出，检查响度值 
    # 如果音频的响度低于阈值，认为音频是静音
    if dBFS < silence_threshold:
        return True
    # 获取音频的样本数据
    samples = np.array(audio.get_array_of_samples())   
    # 计算样本的绝对值均值，如果均值接近零，认为是静音
    avg_sample = np.mean(np.abs(samples))
    print(f"Average sample value: {avg_sample}")  # 调试输出，查看均值   
    # 如果样本均值非常接近零，认为是静音
    if avg_sample < sample_threshold:
        return True   
    return False