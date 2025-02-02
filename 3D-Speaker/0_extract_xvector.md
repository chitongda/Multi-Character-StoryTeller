# Install modelscope
# pip install modelscope
# # ERes2Net trained on 200k labeled speakers
# model_id=iic/speech_eres2net_sv_zh-cn_16k-common
# ERes2NetV2 trained on 200k labeled speakers
model_id=iic/speech_eres2netv2_sv_zh-cn_16k-common
# CAM++ trained on 200k labeled speakers
model_id="iic/speech_campplus_sv_zh-cn_16k-common"
# Run CAM++ or ERes2Net inference
python speakerlab/bin/infer_sv.py --model_id $model_id
# Run batch inference
python speakerlab/bin/infer_sv_batch.py --model_id $model_id --wavs $wav_list

python speakerlab/bin/infer_sv.py --model_id iic/speech_eres2netv2_sv_zh-cn_16k-common

python speakerlab/bin/infer_sv.py --model_id iic/speech_eres2netv2_sv_zh-cn_16k-common --wavs E:\2_PYTHON\Project\StoryTeller\0_downLoad_Audio\2.0_wav_list\1-5.txt --saveFile E:\2_PYTHON\Project\StoryTeller\0_downLoad_Audio\2.1_xvector\

python speakerlab/bin/infer_sv_batch.py --model_id iic/speech_eres2netv2_sv_zh-cn_16k-common --wavs E:\2_PYTHON\Project\StoryTeller\0_downLoad_Audio\2.0_wav_list\1-5.txt --feat_out_dir E:\2_PYTHON\Project\StoryTeller\0_downLoad_Audio\2.1_xvector\

# SDPN trained on VoxCeleb
model_id=iic/speech_sdpn_ecapa_tdnn_sv_en_voxceleb_16k
# Run SDPN inference
python speakerlab/bin/infer_sv_ssl.py --model_id $model_id

# Run RDINO inference
model_id=damo/speech_rdino_ecapa_tdnn_sv_en_voxceleb_16k
python speakerlab/bin/infer_sv_ssl.py --model_id $model_id --yaml egs/voxceleb/sv-rdino/conf/rdino.yaml

# Run diarization inference
python speakerlab/bin/infer_diarization.py --wav [wav_list OR wav_path] --out_dir $out_dir
# Enable overlap detection
python speakerlab/bin/infer_diarization.py --wav [wav_list OR wav_path] --out_dir $out_dir --include_overlap --hf_access_token $hf_access_token