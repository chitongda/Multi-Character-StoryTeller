from transformers import AutoModelForCausalLM, AutoTokenizer
import torchaudio
import pygame
import time
from openai import OpenAI 
import edge_tts
import asyncio
import os
import shutil
from pydub import AudioSegment
import os
from os.path import isfile, join

# --- 播放音频 -
def play_audio(file_path):
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(1)  # 等待音频播放结束
        print("播放完成！")
    except Exception as e:
        print(f"播放失败: {e}")
    finally:
        pygame.mixer.quit()

folder_path = "./out_story/"

def apply_fade(audio, fade_duration=300):
    """
    为音频片段的开头和结尾添加淡入淡出效果。

    :param audio: AudioSegment 输入音频片段
    :param fade_duration: int 淡入淡出的持续时间（毫秒）
    :return: AudioSegment 添加淡入淡出效果后的音频片段
    """
    return audio.fade_in(fade_duration).fade_out(fade_duration)

# 初始化音频片段
combined = AudioSegment.empty()

audio_file_count = 0
target_sample_rate = 22050
silence = AudioSegment.silent(duration=100)  # 0.2秒静音
for idx in range(len(os.listdir(folder_path))):
    # play_audio(f'{folder_path}/story_{audio_file_count}.wav')

    audio = AudioSegment.from_wav(f'{folder_path}/story_{audio_file_count}.wav')

    # 如果需要，重新采样
    # if audio.frame_rate != target_sample_rate:
    #     audio = audio.set_frame_rate(target_sample_rate)

    audio = apply_fade(audio)
    # combined += audio  # 拼接音频片段
    combined += audio + silence  # 拼接音频片段


    audio_file_count += 1

# 导出拼接后的音频文件
output_file = 'combined.wav'
combined.export(output_file, format="wav")
print(f'All WAV files have been combined into {output_file}')