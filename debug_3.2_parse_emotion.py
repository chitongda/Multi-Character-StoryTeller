# 情绪识别，5分类，'生气/angry', '开心/happy', '中立/neutral', '难过/sad', '<unk>'


from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inference_pipeline = pipeline(
    task=Tasks.emotion_recognition,
    model="iic/emotion2vec_plus_large")

dict_emo_label = {
    0:'生气',
    1:'开心',
    2:'中立',
    3:'难过',
    4:'未知'
}

# wav_path = 'https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav'
wav_path = r'E:\2_PYTHON\Project\StoryTeller\0_downLoad_Audio\2_speaker_wav\1-5\2\2_0.wav'
rec_result = inference_pipeline(wav_path, granularity="utterance", extract_embedding=False)
print(rec_result)
pred_scores = rec_result[0]['scores']
max_index = pred_scores.index(max(pred_scores))
pred_emo = dict_emo_label[max_index]
breakpoint()
# 