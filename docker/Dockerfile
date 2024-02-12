FROM nvcr.io/nvidia/pytorch:22.10-py3

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

RUN apt-get -y update
RUN apt install -y tmux
RUN apt install -y htop
RUN apt install -y ncdu

COPY requirements.txt ./
RUN pip install -r requirements.txt && rm ./requirements.txt