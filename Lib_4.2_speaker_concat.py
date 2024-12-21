
import json
from collections import defaultdict
import numpy as np
import os

# --- 修正写json时的格式问题 ---
def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)  # 转换 NumPy 的整数类型为 Python 的 int
    elif isinstance(obj, np.floating):
        return float(obj)  # 转换 NumPy 的浮点类型为 Python 的 float
    else:
        return obj  # 对于其他类型不做处理

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="语音分段截取")
    parser.add_argument(
        "--save_folder", 
        type=str, 
        default='./0_downLoad_Audio/5_parse_profile/3_speaker_wav_new.json',
        help="重整后保存目录"
    )

    parser.add_argument(
        "--file_profile", 
        type=str, 
        default='./0_downLoad_Audio/5_parse_profile/3_speaker_wav.json', 
        help=""
    )

    parser.add_argument(
        "--directory_xvector", 
        type=str, 
        default='./0_downLoad_Audio/4.1_xvector/embeddings', 
        help="声纹xvector.npy保存目录"
    )
    args = parser.parse_args()

    json_speaker_file = args.file_profile
    path_root_npy = args.directory_xvector
    json_speaker_file_new = args.save_folder

    # json_speaker_file = r'E:\2_PYTHON\Project\StoryTeller\0_downLoad_Audio\3_parse_profile\2_speaker_wav.json'
    # path_root_npy = r'E:\2_PYTHON\Project\StoryTeller\0_downLoad_Audio\2.1_xvector\embeddings'
    # json_speaker_file_new = r'E:\2_PYTHON\Project\StoryTeller\0_downLoad_Audio\3_parse_profile\2_speaker_wav_new.json'

    dict_speaker_file = defaultdict(list)

    try:
        with open(json_speaker_file, "r", encoding="utf-8") as file:
            data = json.load(file)  # 解析 JSON 文件内容为 Python 对象
            # breakpoint()
            for key in data.keys():
                speaker_num   = key
                speaker_name  = data[key]['speaker']
                speaker_files = data[key]['files']

                dict_speaker_file[speaker_name] = speaker_files

    except FileNotFoundError:
        print(f"文件 {json_speaker_file} 不存在！")

    dict_speaker_xector = defaultdict()
    for key in dict_speaker_file.keys():
        xvector_key = [0.0]*192
        for file_wav in dict_speaker_file[key]:
            xvector_tmp = np.load(os.path.join(path_root_npy, os.path.basename(file_wav.split(',')[0].replace('.wav', '.npy'))))
            xvector_key += xvector_tmp

        xvector_key /= len(dict_speaker_file[key])
        dict_speaker_xector[key] = xvector_key

    import numpy as np
    from itertools import combinations

    # 计算两个向量的余弦相似度
    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    # 存储相似度大于 0.8 的键值对
    set_thred = 0.85
    similar_pairs = []
    for key1, key2 in combinations(dict_speaker_xector.keys(), 2):  # 两两组合
        sim = cosine_similarity(dict_speaker_xector[key1], dict_speaker_xector[key2])
        if sim > set_thred:
            similar_pairs.append((key1, key2))

    # print("相似度大于 0.8 的键值对：")
    # print(similar_pairs)

    # 融合键值对
    def merge_groups(pairs):
        groups = []
        for key1, key2 in pairs:
            found = False
            for group in groups:
                if key1 in group or key2 in group:
                    group.update([key1, key2])  # 将 key1 和 key2 加入已有组
                    found = True
                    break
            if not found:
                groups.append(set([key1, key2]))  # 创建新组
        return [list(group) for group in groups]

    merged_keys = merge_groups(similar_pairs)

    print("融合后的键值组合：")
    print(len(merged_keys), merged_keys)
    # breakpoint()
    merged_keys_flattened = [item for row in merged_keys for item in row]  # 将二维列表展平为一维列表
    print(len(merged_keys_flattened))

    dict_selected_name_profile = defaultdict(list)

    for idx in range(len(merged_keys)):
        files_temp = []
        list_speaker_name = merged_keys[idx]
        for speaker_name in list_speaker_name:
            files_temp.extend(dict_speaker_file[speaker_name])
        
        dict_selected_name_profile[speaker_name] = files_temp



    with open(json_speaker_file_new, 'w', encoding='utf-8') as f:
        data = {}
        for idx, key in enumerate(dict_selected_name_profile.keys()):
            # 将字典写入JSON文件
            data[f'speaker_{idx}'] = {}
            data[f'speaker_{idx}']['speaker'] = f'{key}'
            data[f'speaker_{idx}']['files'] = dict_selected_name_profile[key]
        
        for idx_1, key in enumerate(dict_speaker_file.keys()):
            if key not in merged_keys_flattened:
                data[f'speaker_{idx+idx_1}'] = {}
                data[f'speaker_{idx+idx_1}']['speaker'] = f'{key}'
                data[f'speaker_{idx+idx_1}']['files'] = dict_speaker_file[key]

        json.dump(data, f, ensure_ascii=False, indent=4, default=convert_numpy_types)
        
