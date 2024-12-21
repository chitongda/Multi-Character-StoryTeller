import os
import json
import numpy as np

# --- 修正写json时的格式问题 ---
def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)  # 转换 NumPy 的整数类型为 Python 的 int
    elif isinstance(obj, np.floating):
        return float(obj)  # 转换 NumPy 的浮点类型为 Python 的 float
    else:
        return obj  # 对于其他类型不做处理

import argparse
parser = argparse.ArgumentParser(description="语音分段截取")
parser.add_argument(
    "--input_story", 
    type=str, 
    default=r'./out_story_txt/3.txt',
    help="大模型进行角色划分"
)

parser.add_argument(
    "--json_speaker_file", 
    type=str, 
    default='./0_downLoad_Audio/5_parse_profile/3_speaker_wav_new.json', 
    help="说话人、对应语料"
)

parser.add_argument(
    "--corpus_json", 
    type=str, 
    default='./0_downLoad_Audio/5_parse_profile_selected/corpus.json', 
    help="语音列表保存地址"
)

parser.add_argument(
    "--role_corpus_json", 
    type=str, 
    default='./0_downLoad_Audio/5_parse_profile_selected/role_corpus.json', 
    help="语音列表保存地址"
)

parser.add_argument(
    "--role_corpus_seted", 
    type=str, 
    default='./0_downLoad_Audio/5_parse_profile_selected/role_corpus_seted.json', 
    help="语音列表保存地址"
)

args = parser.parse_args()

input_story          =  args.input_story
json_speaker_file    =  args.json_speaker_file
corpus_json          =  args.corpus_json
role_corpus_json     =  args.role_corpus_json
role_corpus_seted    =  args.role_corpus_seted

list_role = []
list_speaker = []

# 读取已设定角色
seted_role = []
with open(role_corpus_seted, "r", encoding="utf-8") as file:
    data_seted = json.load(file)  # 解析 JSON 文件内容为 Python 对象
seted_role = list(data_seted.keys())

# 读取大模型处理后待合成文本
inp_lines = open(input_story, 'r', encoding='utf-8').readlines()
for line in inp_lines:
    # 空行不处理
    if not line.strip():
        continue
    role = line.strip().split('：')[0].replace('[','').replace(']', '').split("_")[0]
    if role not in list_role:
        list_role.append(role)

for role_tmp in list_role:
    if role_tmp not in seted_role:
        data_seted[role_tmp] = ''
with open(role_corpus_seted, 'w', encoding='utf-8') as file:
    json.dump(data_seted, file, ensure_ascii=False, indent=4, default=convert_numpy_types)

try:
    with open(json_speaker_file, "r", encoding="utf-8") as file:
        data = json.load(file)  # 解析 JSON 文件内容为 Python 对象
        # breakpoint()
        for key in data.keys():
            speaker_name  = data[key]['speaker']
            list_speaker.append(speaker_name)
except FileNotFoundError:
    print(f"文件 {json_speaker_file} 不存在！")

print(len(list_role), len(list_speaker))
print(list_role)
print(list_speaker)

with open(role_corpus_json, 'w', encoding='utf-8') as f:
    data = {}
    for role in list_role:
        data[role] = ''
    json.dump(data, f, ensure_ascii=False, indent=4, default=convert_numpy_types)

with open(corpus_json, 'w', encoding='utf-8') as f:
    data = {}
    for role in list_speaker:
        data[role] = ''
    json.dump(data, f, ensure_ascii=False, indent=4, default=convert_numpy_types)
