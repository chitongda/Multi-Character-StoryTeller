
from transformers import AutoModelForCausalLM, AutoTokenizer
import torchaudio
# import pygame
# import time
# from openai import OpenAI 
# import edge_tts
import asyncio
import os
import shutil
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from funasr import AutoModel
import torch

from collections import defaultdict
import json
import re
import random

def filter_text(input_text):
    # 定义正则表达式：匹配中文、英文单词、数字、空格、以及特定标点符号
    pattern = r'[\w\u4e00-\u9fff]+|[.,!?\u3002\uff0c\uff01\uff1f]|\s'
    
    # 使用正则表达式提取匹配的部分
    filtered_parts = re.findall(pattern, input_text)
    
    # 将匹配到的部分拼接成新的字符串
    filtered_text = ''.join(filtered_parts)
    
    return filtered_text

def split_text_by_punctuation(text):
    """
    将输入的汉字或英文混杂字符串按照标点符号切分，并在不超过20字的情况下尽可能多的合并片段。

    支持的标点符号包括：中文标点和英文标点。

    :param text: str 输入的字符串
    :return: list[str] 按标点切分并合并后的字符串列表
    """
    # 定义匹配所有常见中英文标点符号的正则表达式
    punctuation_pattern = r'[\.,!?\"\'\;:\-，。]'

    # 使用 re.split 切分文本，保留标点符号
    segments = re.split(f'({punctuation_pattern})', text)

    # 去掉空字符串并合并文本和标点
    filtered_segments = []
    for i in range(len(segments)):
        if segments[i].strip():  # 去除空白部分
            filtered_segments.append(segments[i])

    # 合并片段，尽可能多地拼接不超过20字的部分
    result = []
    temp = ""
    for segment in filtered_segments:
        if len(temp) + len(segment) <= 30:
            temp += segment
        else:
            if temp:
                result.append(temp)
            temp = segment
    if temp:
        result.append(temp)

    return result



def split_by_punctuation(input_text):
    # 定义正则表达式：匹配中英文标点符号并保留分割符
    pattern = r'([，。！？\.!?\uff01\uff1f,\,;\:\-\—])'
    
    # 使用正则表达式切分并保留标点符号
    split_parts = re.split(pattern, input_text)
    
    # 将文本和标点重新组合，去除多余空字符串
    result = []
    for i in range(0, len(split_parts) - 1, 2):
        sentence = split_parts[i].strip() + split_parts[i + 1]
        if sentence:
            result.append(sentence)
    if len(split_parts) % 2 == 1 and split_parts[-1].strip():
        result.append(split_parts[-1].strip())
    
    return result

def split_text(text, max_words=30):
    # 使用正则表达式根据标点符号切分文本
    # 这里匹配的标点符号包括句号、逗号、问号、叹号、分号、冒号
    sentences = re.split(r'([，,.。！？、；：])', text)
    # 将分割的句子重新组合
    sentences = [s.strip() + (sentences[i+1] if i+1 < len(sentences) else '') for i, s in enumerate(sentences) if i % 2 == 0]
    
    # 用来保存最终切分后的文本片段
    result = []
    current_chunk = []
    current_word_count = 0
    
    # breakpoint()

    # 遍历句子并按最大词数分段
    for sentence in sentences:
        word_count = len(sentence)
        
        # 如果当前句子加上当前片段的词数超过限制，就开始新的片段
        if current_word_count + word_count > max_words:
            result.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_word_count = word_count
        else:
            current_chunk.append(sentence)
            current_word_count += word_count
    
    # 添加最后一个片段
    if current_chunk:
        result.append(" ".join(current_chunk))
    
    return result

def clear_folder(folder_path):
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        print(f"文件夹 '{folder_path}' 不存在，已创建")
        return
    
    # 获取文件夹中的所有文件和子文件夹
    items = os.listdir(folder_path)
    
    # 如果文件夹为空，直接返回
    if not items:
        print(f"文件夹 '{folder_path}' 已经为空")
        return
    
    # 遍历文件和文件夹并删除
    for item in items:
        item_path = os.path.join(folder_path, item)
        
        # 判断是否是文件夹或文件
        if os.path.isfile(item_path):
            os.remove(item_path)  # 删除文件
            print(f"删除文件: {item_path}")
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)  # 删除文件夹及其内容
            print(f"删除文件夹: {item_path}")
    
    print(f"文件夹 '{folder_path}' 已清空")


if __name__ == "__main__":
    os.system('set CUDA_LAUNCH_BLOCKING=1')
    
    file_wav_text = r'E:\2_PYTHON\Project\StoryTeller\0_downLoad_Audio\5_parse_profile_selected\wav_asr_len.txt'
    file_inp = r'E:\2_PYTHON\Project\StoryTeller\out_story_txt\3.txt'
    role_corpus = r'E:\2_PYTHON\Project\StoryTeller\0_downLoad_Audio\5_parse_profile_selected\role_corpus_seted.json'
    corpus_library = r'E:\2_PYTHON\Project\StoryTeller\0_downLoad_Audio\5_parse_profile\3_speaker_wav_new.json'
    path_cosyvoice_model = r'E:\2_PYTHON\Project\GPT\QWen\pretrained_models\CosyVoice-300M'
    system_path = 'E:\\2_PYTHON\\Project\\StoryTeller\\0_downLoad_Audio\\'
    
    # cosyvoice = CosyVoice(path_cosyvoice_model, load_jit=True, load_onnx=False, fp16=True)
    cosyvoice2_model_path = r'E:\2_PYTHON\Project\GPT\QWen\pretrained_models\CosyVoice2-0.5B'
    cosyvoice = CosyVoice2(cosyvoice2_model_path, load_jit=True, load_onnx=False, load_trt=False)
    
    # -- 构建 语音名-语音长度的词典 --
    dict_wav_textLen = defaultdict()
    dict_wav_text = defaultdict()
    for line in open(file_wav_text, 'r', encoding='utf-8').readlines():
        wav_name = line.split('    ')[0]
        text = line.strip().split('    ')[1]
        len_text = len(text)
        dict_wav_textLen[wav_name] = len_text
        dict_wav_text[wav_name] = text
    # breakpoint()
    
    dict_role_speaker = defaultdict()
    with open(role_corpus, "r", encoding="utf-8") as file:
        data = json.load(file)
        if isinstance(data, dict):  # 如果 JSON 文件是对象类型
            dict_role_speaker.update(data)
        else:
            print("JSON 文件内容不是字典类型，无法写入 result_dict")
    
    inp_lines = open(file_inp, encoding='utf-8').readlines()

    role_dict = defaultdict(int)
    role_idx = 0
    for line in inp_lines:
        role = line.strip().split('：')[0].replace('[','').replace(']', '').split('_')[0]
        if role not in role_dict.keys():
            role_dict[role] = role_idx
            role_idx += 1
    print(role_dict)

    dict_speaker_wavs = defaultdict(list)
    with open(corpus_library, 'r', encoding='utf-8') as f:
        a = json.load(f)
        for key in a.keys():
            speaker_profile = a[key]['speaker']
            wavs_list = a[key]['files']
            dict_speaker_wavs[speaker_profile] = wavs_list
        # breakpoint()

    # -------- SenceVoice 语音识别 --模型加载-----
    # model_dir = r"E:\2_PYTHON\Project\GPT\QWen\pretrained_models\SenseVoiceSmall"
    # model_senceVoice = AutoModel( model=model_dir, trust_remote_code=True)

    folder_path = "./out_story/"
    clear_folder(folder_path)

    total_num = 0
    for line in inp_lines:
        # breakpoint()
        try:
            role = line.strip().split('：')[0].replace('[','').replace(']', '').split('_')[0]
            text = line.strip().split('：')[1].replace('“','').replace('”', '')
            text = filter_text(text)

            role_profile = dict_role_speaker[role]
            role_speaker = random.choice(role_profile)
            prompt_files = dict_speaker_wavs[role_speaker]
            # breakpoint()
            len_list = []
            for prompt_tmp in prompt_files:
                prompt_wav = prompt_tmp.split(',')[0]
                len_txt = dict_wav_textLen[prompt_wav]
                # len_txt = dict_wav_textLen[prompt_wav.replace('./0_downLoad_Audio/', f'{system_path}')]
                len_list.append(len_text)
            # print(len_list)
            closest_index = min(range(len(len_list)), key=lambda i: abs(len_list[i] - len(text)))

            prompt_wav = prompt_files[closest_index].split(',')[0]
            prompt_text = dict_wav_text[prompt_wav]
            print(prompt_text)

            # try:
            #     prompt_wav = random.choice(prompt_files).split(',')[0]
            # except:
            #     prompt_wav = prompt_files[0].split(',')[0]

            # breakpoint()
            # res = model_senceVoice.generate(
            #         input=prompt_wav,
            #         cache={},
            #         language="auto", # "zn", "en", "yue", "ja", "ko", "nospeech"
            #         use_itn=False,
            #     )
            # prompt_text = res[0]['text'].split(">")[-1]
            # print(prompt_text)

            
            prompt_speech_16k = load_wav(prompt_wav, 16000)
            
            max_word = 20
            try:
                if len(text) > max_word:
                    # texts = split_by_punctuation(text)
                    # texts = split_text_by_punctuation(text)
                    texts = split_text(text, max_words=max_word)
                    
                    for i_num, text_tmp in enumerate(texts):
                        print("--split:--", i_num, text_tmp)
                        for i, j in enumerate(cosyvoice.inference_zero_shot(text_tmp.replace('。', ''), prompt_text, prompt_speech_16k, stream=False)):
                            torchaudio.save('{}/story_{}.wav'.format(folder_path,total_num), j['tts_speech'], 22050)
                            total_num += 1
                else:
                    for i, j in enumerate(cosyvoice.inference_zero_shot(text.replace('。', ''), prompt_text, prompt_speech_16k, stream=False)):
                        torchaudio.save('{}/story_{}.wav'.format(folder_path,total_num), j['tts_speech'], 22050)
                        total_num += 1
            except:
                continue
        except:
            continue
        