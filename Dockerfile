FROM nvidia/cuda:9.0-base
MAINTAINER Jules bamboo.cutecat@gmail.com 


### install conda env
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh\
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version

RUN conda create --name pysot python=3.5 -y
ENV PATH /opt/conda/envs/pysot/bin:$PATH
# 開啟root shell
SHELL [ "/bin/bash", "--login", "-c" ]                                     


## dependencies
RUN source activate pysot\
&& conda install numpy \
&& pip install tensorboardX \
&& pip install pyyaml yacs tqdm colorama matplotlib cython

#opencv dependencies
RUN source activate pysot\
&& apt-get update\
&& pip install opencv-python\
&& apt-get install libgtk2.0-dev -y\
&& apt-get install -y libsm6 libxext6\
&& apt-get install -y libxrender-dev\
&& conda install -y pytorch=0.4.1 torchvision cuda90 -c pytorch

# gstreamer dependencies
RUN apt-get install python3-gi -y
RUN ln -s /usr/lib/python3/dist-packages/gi /root/miniconda3/envs/pysot/lib/python3.5/site-packages/
RUN apt-get install -y libgstreamer1.0-0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-doc gstreamer1.0-tools


## ADD code
ADD /tracking/ /pysot/
ENV PYTHONPATH="/pysot:$PYTHONPATH"
WORKDIR /pysot
RUN source activate pysot\
&& python setup.py build_ext --inplace 

RUN source activate pysot\
&&  pip install pyserial


## env set up
################################################################
# distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
# curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
# curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
# sudo systemctl restart docker
# docker run --gpus all nvidia/cuda:10.0-base nvidia-smi


# sudo groupadd docker
# sudo usermod -aG docker $USER

# xhost +local:root
# docker run -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY --device=/dev/video0:/dev/video0 -it pysot /bin/bash

# docker run --gpus all -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY --device=/dev/video0:/dev/video0 -it pysot /bin/bash
################################################################