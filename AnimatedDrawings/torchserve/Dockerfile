# syntax = docker/dockerfile:1.2

FROM continuumio/miniconda3:24.1.2-0

# install os dependencies
RUN mkdir -p /usr/share/man/man1
RUN sed -i "s|^deb http://deb.debian.org/debian bullseye main|deb [check-valid-until=no] http://snapshot.debian.org/archive/debian/20240211T000000Z bullseye main|; s|^deb http://deb.debian.org/debian-security bullseye-security main|deb [check-valid-until=no] http://snapshot.debian.org/archive/debian-security/20240211T000000Z bullseye-security main|; s|^deb http://deb.debian.org/debian bullseye-updates main|deb [check-valid-until=no] http://snapshot.debian.org/archive/debian/20240211T000000Z bullseye-updates main|" /etc/apt/sources.list && apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    ca-certificates \
    curl \
    vim \
    sudo \
    default-jre \
    git \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN conda install python=3.8.13 -y


# install python dependencies
RUN pip install openmim
RUN pip install torch==2.0.0
RUN mim install mmcv-full==1.7.0
RUN pip install mmdet==2.27.0
RUN pip install torchserve

# bugfix for xtcocoapi, an mmpose dependency
RUN git clone https://github.com/jin-s13/xtcocoapi.git
WORKDIR xtcocoapi
RUN pip install -r requirements.txt
RUN python setup.py install
WORKDIR /
RUN pip install mmpose==0.29.0
RUN pip install torchvision==0.15.1  # solve torch version problem
RUN pip install numpy==1.24.4  #solve numpy version problem

# prep torchserve
RUN mkdir -p /home/torchserve/model-store
RUN wget https://github.com/facebookresearch/AnimatedDrawings/releases/download/v0.0.1/drawn_humanoid_detector.mar -P /home/torchserve/model-store/
RUN wget https://github.com/facebookresearch/AnimatedDrawings/releases/download/v0.0.1/drawn_humanoid_pose_estimator.mar -P /home/torchserve/model-store/
COPY config.properties /home/torchserve/config.properties

# starting command
CMD /opt/conda/bin/torchserve --start --disable-token-auth --ts-config /home/torchserve/config.properties && sleep infinity
