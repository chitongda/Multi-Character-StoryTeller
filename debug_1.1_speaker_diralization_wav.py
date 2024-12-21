# 基于阿里开源eres2net流程实现说话人分离
# 模型会自动下载
# 依赖hdbscan库：pip install hdbscan
# 更新funasr库：pip install -U funasr, funasr==1.1.12

import os
import json
import numpy as np
from modelscope.pipelines import pipeline

# --- 修正写json时的格式问题 ---
def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)  # 转换 NumPy 的整数类型为 Python 的 int
    elif isinstance(obj, np.floating):
        return float(obj)  # 转换 NumPy 的浮点类型为 Python 的 float
    else:
        return obj  # 对于其他类型不做处理
#  ----------------------------

# --- 分离流程 ---
sd_pipeline = pipeline(
    task='speaker-diarization',
    model='damo/speech_eres2net-large_speaker-diarization_common',
    model_revision='v1.0.0'
)

save_result = './0_downLoad_Audio/1_speaker_diralization_result/'
input_wav = r'E:\2_PYTHON\Project\StoryTeller\0_downLoad_Audio\processed\多人有声剧《斗破苍穹》完整版 p01 01-05.wav'
save_file = os.path.join(save_result, os.path.basename(input_wav).replace('.wav', '.json'))

result = sd_pipeline(input_wav)
print(result)


with open(save_file, "w") as f:
    # json.dump(result, f, indent=4, ensure_ascii=False)
    json.dump(result, f, ensure_ascii=False, default=convert_numpy_types)


# 如果有先验信息，输入实际的说话人数，会得到更准确的预测结果
# result = sd_pipeline(input_wav, oracle_num=2)
# print(result)
