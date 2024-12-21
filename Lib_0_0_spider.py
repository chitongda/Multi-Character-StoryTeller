# 基于yt-dlp库：pip install -U yt-dlp 安装
# 确保安装ffmpeg，如无，先安装ffmpeg
# 验证是否安装成功：yt-dlp --version
# 2024.12.13

import os
from pydub import AudioSegment
import subprocess

def download_audio(bilibili_url, output_dir):
    """
    使用 yt-dlp 从 Bilibili 爬取音频并保存为原始文件。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 下载音频命令
    command = [
        "yt-dlp", 
        "-x", "--audio-format", "mp3",  # 只提取音频并保存为 MP3
        "-o", f"{output_dir}/%(title)s.%(ext)s",  # 输出文件名
        bilibili_url
    ]
    
    try:
        print(f"正在下载 {bilibili_url} 的音频...")
        subprocess.run(command, check=True)
        print(f"音频下载完成，保存至 {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"下载失败：{e}")

def process_audio(input_dir, output_dir, target_sample_rate=16000):
    """
    批量处理音频文件，将其转换为单通道并重新采样为目标采样率。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取所有音频文件
    audio_files = [f for f in os.listdir(input_dir) if f.endswith(('.mp3', '.wav'))]
    
    if not audio_files:
        print(f"目录 {input_dir} 中没有音频文件！")
        return

    for audio_file in audio_files:
        input_path = os.path.join(input_dir, audio_file)
        output_path = os.path.join(output_dir, os.path.splitext(audio_file)[0] + ".wav")
        
        try:
            # 读取音频文件
            print(f"正在处理音频文件：{audio_file}")
            audio = AudioSegment.from_file(input_path)
            
            # 转换为单通道
            audio = audio.set_channels(1)
            
            # 重新采样
            audio = audio.set_frame_rate(target_sample_rate)
            
            # 保存为 WAV 格式
            audio.export(output_path, format="wav")
            print(f"处理完成，保存为 {output_path}")
        except Exception as e:
            print(f"处理文件 {audio_file} 时出错：{e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="从 Bilibili 爬取音频并处理为单通道 WAV 格式")
    parser.add_argument(
        "--url", 
        type=str, 
        # default='https://www.bilibili.com/video/BV1khmtYGEvk?vd_source=9c971a6f4a09f2e118bed0d43bdd8e18&spm_id_from=333.788.videopod.episodes',
        default='https://www.bilibili.com/video/BV1aEz5YvESD/?spm_id_from=333.337.search-card.all.click&vd_source=9c971a6f4a09f2e118bed0d43bdd8e18',
        help="Bilibili 视频的 URL"
    )
    parser.add_argument(
        "--download_dir", 
        type=str, 
        default="./0_downLoad_Audio/0_downloads", 
        help="音频下载保存的目录，默认为 'downloads'"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./0_downLoad_Audio/1_processed", 
        help="处理后音频保存的目录，默认为 'processed'"
    )
    parser.add_argument(
        "--sample_rate", 
        type=int, 
        default=16000, 
        help="目标采样率，默认为 16000 Hz"
    )
    
    args = parser.parse_args()
    
    try:
        # 下载音频
        # download_audio(args.url, args.download_dir)
        
        # 处理音频
        process_audio(args.download_dir, args.output_dir, args.sample_rate)
    except Exception as e:
        print(f"发生错误：{e}")
