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


# 初始化说话人分离模型
sd_pipeline = pipeline(
    task='speaker-diarization',
    model='damo/speech_eres2net-large_speaker-diarization_common',
    model_revision='v1.0.0'
)

AUDIO_EXTENSIONS = ('.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="说话人分离")
    parser.add_argument(
        "--input_folder", 
        type=str, 
        default='./0_downLoad_Audio/1_processed/',
        help="默认下载后处理为16K-16bit的默认音频目录。如文件路径输入，加双引号"
    )
    
    parser.add_argument(
        "--save_result", 
        type=str, 
        default="./0_downLoad_Audio/2_speaker_diarization_result", 
        help="说话人分离结果，默认为 'downloads'"
    )
    args = parser.parse_args()

    # 设置输入文件夹和输出文件夹
    input_folder = args.input_folder
    save_result = args.save_result

    print(f'待处理文件为：{input_folder}')
    print(f'分离结果保存目录为：{save_result}')

    # 确保输出文件夹存在
    os.makedirs(save_result, exist_ok=True)

    if os.path.isdir(input_folder):
        # 遍历文件夹中的每个音频文件
        for file_name in os.listdir(input_folder):
            if file_name.endswith('.wav'):
                input_wav = os.path.join(input_folder, file_name)
                save_file = os.path.join(save_result, file_name.replace('.wav', '.json'))

                # 执行说话人分离
                print(f"Processing: {input_wav}")
                result = sd_pipeline(input_wav)
                print(result)

                # 保存结果到JSON文件
                with open(save_file, "w") as f:
                    json.dump(result, f, ensure_ascii=False, default=convert_numpy_types)

                print(f"Saved result to: {save_file}")

    elif os.path.isfile(input_folder):
        # 获取文件扩展名并判断是否是语音文件
        _, ext = os.path.splitext(input_folder)
        if ext.lower() in AUDIO_EXTENSIONS:
            input_wav = input_folder
            save_file = os.path.join(save_result, os.path.basename(input_wav).replace('.wav', '.json'))

            # 执行说话人分离
            print(f"Processing: {input_wav}")
            result = sd_pipeline(input_wav)
            print(result)

            # 保存结果到JSON文件
            with open(save_file, "w") as f:
                json.dump(result, f, ensure_ascii=False, default=convert_numpy_types)

            print(f"Saved result to: {save_file}")

    print("Processing complete.")
