
import os
import json
import numpy as np
from modelscope.pipelines import pipeline
from pydub import AudioSegment
from collections import defaultdict

# --- 输入 起始、结束 时间点，截取音频，保存至指定目录 ---
def cut_audio(input_path, start_time, end_time, output_path):
    """
    截取 WAV 文件的指定时间段，并保存到指定目录。
    
    :param input_path: 输入的 WAV 文件路径
    :param start_time: 起始时间（秒）
    :param end_time: 结束时间（秒）
    :param output_path: 截取后保存的新文件路径
    """
    # 加载音频文件
    audio = AudioSegment.from_wav(input_path)
    
    # 将时间转换为毫秒
    start_ms = start_time * 1000  # 秒转毫秒
    end_ms = end_time * 1000      # 秒转毫秒
    
    # 截取音频文件，大于10秒截断
    if end_time-start_time > 10:
        end_ms = (start_time+10) * 1000
    cut_audio = audio[start_ms:end_ms]
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存截取后的音频文件
    cut_audio.export(output_path, format="wav")
    print(f"音频已保存至: {output_path}")


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description="语音分段截取")
    parser.add_argument(
        "--input_folder", 
        type=str, 
        default='./0_downLoad_Audio/1_processed/',
        help="默认下载后处理为16K-16bit的默认音频目录。如文件路径输入，加双引号"
    )
    
    parser.add_argument(
        "--speaker_json", 
        type=str, 
        default="./0_downLoad_Audio/2_speaker_diarization_result", 
        help="说话人分离结果，默认为 'downloads'"
    )

    parser.add_argument(
        "--save_result", 
        type=str, 
        default="./0_downLoad_Audio/3_speaker_wav", 
        help="根据分离结果，保存每个片段"
    )
    args = parser.parse_args()

    input_folder = args.input_folder
    speaker_json = args.speaker_json
    save_cut_wav_root = args.save_result

    # 遍历输入文件夹中语音文件
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.wav'):
            try:
                # 匹配 语音名、对应分离后json文件
                input_wav = os.path.join(input_folder, file_name)
                json_file = os.path.join(speaker_json, os.path.basename(input_wav).replace('.wav', '.json'))

                if not os.path.isfile(json_file):
                    continue
                
                name_file = os.path.basename(input_wav).replace('.wav', '')
                os.makedirs(save_cut_wav_root, exist_ok=True)
                dict_num = defaultdict(int)

                #  json文件解析，获取每段音频start\end-time, 说话人编号
                for line in open(json_file).readlines():
                    a = json.loads(line.strip())
                    num_sentence = len(a['text'])
                    
                    for i in range(num_sentence):
                        start = a['text'][i][0]
                        end = a['text'][i][1]
                        speaker = a['text'][i][2]
                        print(start, end, speaker)

                        # 小于3秒删除，大于10秒的截断，仅取前10秒
                        duration = end - start
                        if duration < 3:
                            continue
                        else:
                            processed_file = os.path.basename(input_wav).replace('.wav', '')
                            save_dir = os.path.join(save_cut_wav_root, f'{processed_file}/{speaker}')
                            os.makedirs(save_dir, exist_ok=True)
                            
                            # 按照 语音名_说话人编号格式保存
                            output_file = os.path.join(save_dir, f'{name_file}_{speaker}_{dict_num[speaker]}.wav')
                            cut_audio(input_wav, start, end, output_file)
                            dict_num[speaker] += 1
            except:
                print("发生了一些异常，请检查输入")
                continue

