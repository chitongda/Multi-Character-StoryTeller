
import os
from funasr import AutoModel

os.system('set CUDA_LAUNCH_BLOCKING=1')

import argparse
parser = argparse.ArgumentParser(description="获取语音目录")
parser.add_argument(
    "--input_folder", 
    type=str, 
    default=r'./0_downLoad_Audio/4.0_wav_list/3_speaker_wav.txt',
    help="切分后音频列表"
)

parser.add_argument(
    "--save_result", 
    type=str, 
    default="./0_downLoad_Audio/5_parse_profile_selected/wav_asr_len.txt", 
    help="根据分离结果，保存每个片段"
)
args = parser.parse_args()

wav_list = args.input_folder
save_result = args.save_result

save_file = open(save_result,'w',encoding='utf-8')

# -------- SenceVoice 语音识别 --模型加载-----
model_dir = r"E:\2_PYTHON\Project\GPT\QWen\pretrained_models\SenseVoiceSmall"
model_senceVoice = AutoModel( model=model_dir, trust_remote_code=True)

for line in open(wav_list, 'r', encoding='utf-8').readlines():
    wav_path = line.strip()

    res = model_senceVoice.generate(
                    input=wav_path,
                    cache={},
                    language="auto", # "zn", "en", "yue", "ja", "ko", "nospeech"
                    use_itn=False,
                )
    prompt_text = res[0]['text'].split(">")[-1]

    save_line = f'{wav_path}    {prompt_text}\n'
    # print(save_line)
    save_file.write(save_line)
