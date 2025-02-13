# 使用基于 Ubuntu 的 Python 3.10 官方镜像作为基础
FROM python:3.10

# 设置工作目录
WORKDIR /app

# 更新系统并安装系统级依赖，包含 portaudio19-dev
RUN apt-get update && apt-get install -y \
    ffmpeg \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# 安装 Anaconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

# 将 Anaconda 添加到环境变量
ENV PATH="/opt/conda/bin:${PATH}"

# 创建虚拟环境
RUN conda create -n chatAudio python=3.10

# 激活虚拟环境并设置环境变量
ENV CONDA_DEFAULT_ENV=chatAudio
ENV CONDA_PREFIX=/opt/conda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH

# 安装 PyTorch + CUDA 版本
RUN pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118

# 简易版本安装，不使用 cosyvoice 时的依赖项
RUN pip install edge-tts==6.1.17 funasr==1.1.12 ffmpeg==1.4 opencv-python==4.10.0.84 transformers==4.45.2 webrtcvad==2.0.10 qwen-vl-utils==0.0.8 pygame==2.6.1 langid==1.1.6 langdetect==1.0.9 accelerate==0.33.0 PyAudio==0.2.14

# 安装 cosyvoice 依赖库
RUN conda install -c conda-forge pynini=2.1.6
RUN pip install WeTextProcessing --no-deps

# 安装 cosyvoice 其它依赖项
RUN pip install HyperPyYAML==1.2.2 modelscope==1.15.0 onnxruntime==1.19.2 openai-whisper==20231117 importlib_resources==6.4.5 sounddevice==0.5.1 matcha-tts==0.0.7.0

# 复制项目文件到工作目录
COPY . .

# 暴露端口（如果项目需要特定端口，这里假设为 50000，可按需修改）
EXPOSE 50000

# 定义容器启动时执行的命令，这里以不调用 cosyvoice 的验证脚本为例
CMD ["python", "13_SenceVoice_QWen2.5_edgeTTS_realTime.py"]