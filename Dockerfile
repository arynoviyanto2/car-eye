# Copyright Ary Noviyanto 2021

FROM ubuntu:21.04
ENV DEBIAN_FRONTEND=noninteractive 
RUN apt update \
    && apt install -y python3-dev python3-pip
RUN apt install libgl1-mesa-glx -y
RUN apt-get install libglib2.0-0 -y
RUN apt install curl -y
RUN apt install unzip -y

RUN pip3 install virtualenv

RUN mkdir root/src


RUN virtualenv -p python3 root/src/.env
RUN /bin/bash -c "source root/src/.env/bin/activate"

COPY . root/src

RUN cd root/src/frames \
    && curl -O https://www.jpjodoin.com/urbantracker/dataset/sherbrooke/sherbrooke_frames.zip \
    && unzip sherbrooke_frames.zip


RUN pip3 install -r root/src/requirements.txt

RUN cd root/src && python3 main.py

WORKDIR root/src

#RUN pip3 install pipenv

#RUN mkdir root/src

#COPY . root/src

#RUN cd root/src && pipenv install --python /usr/bin/python3
#--system --deploy --ignore-pipfile
#RUN cd root/src && pipenv run python3 main.py  

