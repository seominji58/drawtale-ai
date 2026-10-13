# syntax = docker/dockerfile:1.2
#
# Meta AnimatedDrawings 모델을 서빙하는 TorchServe 이미지.
# Meta 원본(Dockerfile.meta-original)에서 두 가지를 바꿨다.
#   1. Debian bullseye 저장소가 만료되어 snapshot 저장소로 고정 (A1 수정)
#   2. PyTorch를 CPU 전용으로 설치 — 배포 대상 Azure VM에 GPU가 없다.
#      기본 설치는 CUDA 라이브러리를 4.5GB 더 받아 이미지가 32GB까지 커진다.

FROM continuumio/miniconda3:24.1.2-0

# install os dependencies
RUN mkdir -p /usr/share/man/man1
RUN sed -i "s|^deb http://deb.debian.org/debian bullseye main|deb [check-valid-until=no] http://snapshot.debian.org/archive/debian/20240211T000000Z bullseye main|; s|^deb http://deb.debian.org/debian-security bullseye-security main|deb [check-valid-until=no] http://snapshot.debian.org/archive/debian-security/20240211T000000Z bullseye-security main|; s|^deb http://deb.debian.org/debian bullseye-updates main|deb [check-valid-until=no] http://snapshot.debian.org/archive/debian/20240211T000000Z bullseye-updates main|" /etc/apt/sources.list && apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    ca-certificates \
    curl \
    wget \
    sudo \
    default-jre \
    git \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN conda install python=3.8.13 -y

# install python dependencies (CPU-only wheels: no CUDA, no triton)
ENV PIP_NO_CACHE_DIR=1
RUN pip install openmim
RUN pip install torch==2.0.0 torchvision==0.15.1 \
    --index-url https://download.pytorch.org/whl/cpu
RUN mim install mmcv-full==1.7.0
RUN pip install mmdet==2.27.0
RUN pip install torchserve

# bugfix for xtcocoapi, an mmpose dependency
RUN git clone https://github.com/jin-s13/xtcocoapi.git
WORKDIR xtcocoapi
RUN pip install -r requirements.txt
RUN python setup.py install
WORKDIR /
RUN rm -rf /xtcocoapi
RUN pip install mmpose==0.29.0
RUN pip install numpy==1.24.4  #solve numpy version problem

# prep torchserve
RUN mkdir -p /home/torchserve/model-store
RUN wget https://github.com/facebookresearch/AnimatedDrawings/releases/download/v0.0.1/drawn_humanoid_detector.mar -P /home/torchserve/model-store/
RUN wget https://github.com/facebookresearch/AnimatedDrawings/releases/download/v0.0.1/drawn_humanoid_pose_estimator.mar -P /home/torchserve/model-store/
COPY config.properties /home/torchserve/config.properties

# starting command
CMD /opt/conda/bin/torchserve --start --disable-token-auth --ts-config /home/torchserve/config.properties && sleep infinity
