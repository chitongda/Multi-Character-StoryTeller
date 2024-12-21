import glob
import os

import argparse
parser = argparse.ArgumentParser(description="语音分段截取")
parser.add_argument(
    "--input_folder", 
    type=str, 
    default=r'E:\2_PYTHON\Project\StoryTeller\0_downLoad_Audio\3_speaker_wav',
    help="切分后音频目录"
)

parser.add_argument(
    "--save_wavList", 
    type=str, 
    default='./0_downLoad_Audio/4.0_wav_list/', 
    help="语音列表保存地址"
)

parser.add_argument(
    "--save_xvector", 
    type=str, 
    default='./0_downLoad_Audio/4.1_xvector/', 
    help="语音列表保存地址"
)

args = parser.parse_args()

input_dir = args.input_folder
save_root = args.save_wavList
os.makedirs(save_root, exist_ok=True)

list_file = files = glob.glob(os.path.join(input_dir, '**', '*'), recursive=True)
save_file =os.path.join(save_root, os.path.basename(input_dir)+'.txt')
save_file_write = open(save_file, 'w', encoding='utf-8')

for file_tmp in list_file:
    if not file_tmp.endswith('.wav'):
        continue
    else:
        line_out = f'{file_tmp}'
        # print(line_out)
        save_file_write.write(line_out + '\n')

# -- 提取声纹
cmd_line = f'python 3D-Speaker/speakerlab/bin/infer_sv.py --model_id iic/speech_eres2netv2_sv_zh-cn_16k-common --wavs {save_file} --saveFile {args.save_xvector}'
os.system(cmd_line)