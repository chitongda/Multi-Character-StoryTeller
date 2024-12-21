
import numpy as np
from pydub import AudioSegment
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

import glob
import os

device = 'cuda'
# 0-----------------  性别、年龄 识别 --------------
class ModelHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config, num_labels):

        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):

        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class AgeGenderModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)
        self.init_weights()

    def forward(
            self,
            input_values,
    ):

        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits_age = self.age(hidden_states)
        logits_gender = torch.softmax(self.gender(hidden_states), dim=1)

        return hidden_states, logits_age, logits_gender
#  --------------------------------------------------------------------------------

# --- 情绪识别模型加载 ---
inference_pipeline_emo = pipeline(task=Tasks.emotion_recognition, model="iic/emotion2vec_plus_large")

# ------------- 性别、年龄 模型 加载 及 推理函数 ---------
model_name = 'audeering/wav2vec2-large-robust-24-ft-age-gender'
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = AgeGenderModel.from_pretrained(model_name).to(device)

def process_func(
    x: np.ndarray,
    sampling_rate: int,
    embeddings: bool = False,
) -> np.ndarray:
    r"""Predict age and gender or extract embeddings from raw audio signal."""

    # run through processor to normalize signal
    # always returns a batch, so we just get the first entry
    # then we put it on the device
    y = processor(x, sampling_rate=sampling_rate)
    y = y['input_values'][0]
    y = y.reshape(1, -1)
    y = torch.from_numpy(y).to(device)

    # run through model
    with torch.no_grad():
        y = model(y)
        if embeddings:
            y = y[0]
        else:
            y = torch.hstack([y[1], y[2]])

    # convert to numpy
    y = y.detach().cpu().numpy()

    return y
# --------------------------------------------------

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description="语音分段截取")
    parser.add_argument(
        "--input_folder", 
        type=str, 
        default=r'E:\2_PYTHON\Project\StoryTeller\0_downLoad_Audio\3_speaker_wav',
        help="切分后音频目录"
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
        default="./0_downLoad_Audio/5_parse_profile", 
        help="根据分离结果，保存每个片段"
    )
    args = parser.parse_args()

    dict_emo_label      = { 0:'生气', 1:'开心', 2:'中立', 3:'难过', 4:'未知'}
    dict_gender_label   = { 0:'女', 1:'男', 2:'儿童'}

    input_dir = args.input_folder
    save_result_root = args.save_result

    list_file = glob.glob(os.path.join(input_dir, '**', '*'), recursive=True)
    # --- 文件保存地址 ---
    os.makedirs(save_result_root, exist_ok=True)
    save_file = open(os.path.join(save_result_root, os.path.basename(input_dir)+'.txt'), 'w', encoding='utf-8')
        
    for file_tmp in list_file:
        if not file_tmp.endswith('.wav'):
            continue

        input_path = file_tmp
        # -------- 性别、年龄 推理 ------------------
        #    Age        female     male       child
        # ------------------------------------------
        sampling_rate = 16000
        audio = AudioSegment.from_wav(input_path)
        signal = np.array(audio.get_array_of_samples()).astype(np.float32)
        result_age_gender = process_func(signal, sampling_rate)
        # breakpoint()
        resulr_gender = result_age_gender[0][1:].tolist()
        pred_gender = dict_gender_label[resulr_gender.index(max(resulr_gender))]
        pred_age = int(100 * result_age_gender[0].tolist()[0])
        

        # --------- 情绪识别 推理 -------------
        rec_result = inference_pipeline_emo(input_path, granularity="utterance", extract_embedding=False)
        # print(rec_result)
        pred_scores = rec_result[0]['scores']
        max_index = pred_scores.index(max(pred_scores))
        pred_emo = dict_emo_label[max_index]

        line_out = f'{file_tmp}, {pred_gender}, {pred_age}, {pred_emo}'
        print(line_out)
        save_file.write(line_out + '\n')
