import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
import os
import glob
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform
import json

# --- 修正写json时的格式问题 ---
def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)  # 转换 NumPy 的整数类型为 Python 的 int
    elif isinstance(obj, np.floating):
        return float(obj)  # 转换 NumPy 的浮点类型为 Python 的 float
    else:
        return obj  # 对于其他类型不做处理
#  ----------------------------

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="语音分段截取")
    parser.add_argument(
        "--save_folder", 
        type=str, 
        default='./0_downLoad_Audio/5_parse_profile/3_speaker_wav.json',
        help="说话人信息保存json"
    )

    parser.add_argument(
        "--file_profile", 
        type=str, 
        default='./0_downLoad_Audio/5_parse_profile/3_speaker_wav.txt', 
        help="语音列表保存地址"
    )

    parser.add_argument(
        "--directory_xvector", 
        type=str, 
        default='./0_downLoad_Audio/4.1_xvector/embeddings', 
        help="声纹xvector.npy保存目录"
    )
    args = parser.parse_args()

    
    directory = args.directory_xvector
    file_profile = args.file_profile
    file_save_profile = args.save_folder

    # --- 构建语音名、属性词典 ---
    dict_name_profile = defaultdict()
    lines = open(file_profile, encoding='utf-8').readlines()
    for line in lines:
        line_array = line.strip().split(', ')
        name = os.path.basename(line_array[0]).split('.')[0]
        dict_name_profile[name] = line.strip()

    # --- 
    dict_selected_name_profile = defaultdict(list)
    dict_speaker = defaultdict(list)

    # 获取目录中所有的.npy文件
    file_paths = glob.glob(os.path.join(directory, '*.npy'))
    for file_tmp in file_paths:
        # 文件名规整，可自定义
        speaker = '_'.join(os.path.basename(file_tmp).split('_')[:-1]).replace(' ', '-').replace('《', '').replace('》', '')
        dict_speaker[speaker].append(file_tmp)

        print(speaker)
        # if '斗破苍穹' in file_tmp:
        #     breakpoint()

    for key in dict_speaker.keys():
        # 初始化一个空列表来存储所有声纹特征和对应的文件名
        features = []
        file_names = []

        # 遍历文件并加载数据
        for file_path in dict_speaker[key]:
            feature = np.load(file_path)
            features.append(feature)
            file_names.append(os.path.basename(file_path))  # 获取文件名

        # 将列表转换为numpy数组
        features = np.vstack(features)
        # 检查特征是否为空或全零
        if features.size == 0 or np.all(features == 0):
            raise ValueError("特征数据为空或全为零，无法计算余弦距离。")

        # 使用欧氏距离进行层次聚类
        # Z = linkage(features, method='ward', metric='euclidean')
        # fixed_threshold = 0.8  # 这里你可以设置一个合适的阈值
        # cluster_labels = fcluster(Z, fixed_threshold, criterion='distance')

        try:
            # 使用余弦距离进行层次聚类
            cosine_distances = pdist(features, metric='cosine')
            # 将距离转换为方阵形式
            square_distances = squareform(cosine_distances)
            # 进行层次聚类
            Z = linkage(square_distances, method='ward')
            # fixed_threshold = 0.8  # 这里你可以设置一个合适的阈值
            # # 根据固定的阈值进行聚类
            # cluster_labels = fcluster(Z, fixed_threshold, criterion='distance')

            # 设置聚类的最大类别数
            max_clusters = 3
            # 根据最大类别数进行聚类
            cluster_labels = fcluster(Z, max_clusters, criterion='maxclust')

            print(key, cluster_labels)
            # 将文件名和聚类标签配对
            file_clusters = dict(zip(file_names, cluster_labels))

            # 计算每个类别的数量
            cluster_counts = {}
            for file_name, cluster_label in file_clusters.items():
                cluster_counts[cluster_label] = cluster_counts.get(cluster_label, 0) + 1

            # 找出类别最多的索引
            max_cluster_index = max(cluster_counts, key=cluster_counts.get)

            # 输出数量最多类对应的文件名
            # print('类别最多的索引是: ', max_cluster_index)
            # print('属于这个类别的文件有：')
            for file_name, cluster_label in file_clusters.items():
                if cluster_label == max_cluster_index:
                    # print(file_name)
                    name = file_name.split('.')[0]
                    # speaker = name.split('_')[0]
                    speaker = '_'.join(os.path.basename(name).split('_')[:-1]).replace(' ', '-').replace(' ', '-').replace('《', '').replace('》', '')
                    wav_name_attribute = dict_name_profile[name]
                    dict_selected_name_profile[speaker].append(wav_name_attribute)

        except:
            continue

    with open(file_save_profile, 'w', encoding='utf-8') as f:
        data = {}
        for idx, key in enumerate(dict_selected_name_profile.keys()):

            age_mean = 0
            for profile_tmp in dict_selected_name_profile[key]:
                age_mean += int(profile_tmp.strip().split(', ')[2])

            age_mean = int(age_mean/len(dict_selected_name_profile[key]))
            profile_ = profile_tmp.strip().split(', ')[1] +'_'+ str(age_mean)
            print(f'{key}_{profile_}')

            # 将字典写入JSON文件
            data[f'speaker_{idx}'] = {}
            data[f'speaker_{idx}']['speaker'] = f'{key}_{profile_}'
            data[f'speaker_{idx}']['files'] = dict_selected_name_profile[key]

        json.dump(data, f, ensure_ascii=False, indent=4, default=convert_numpy_types)
            # json.dump('\n', f)

