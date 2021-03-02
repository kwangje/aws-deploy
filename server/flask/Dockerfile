# Pull tensorflow image with Python3
# FROM tensorflow/tensorflow:2.1.0-py3
# FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel
FROM tiangolo/uwsgi-nginx:python3.7

RUN pip install flask

# Set the working directory to /app
WORKDIR /app

# Transfer content from current dir to /app in container
ADD . /app

# Install audio libraries
RUN apt-get update && apt-get install -y libsndfile1 libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg 

# RUN conda install -c conda-forge uwsgi -y

# Install python packages
RUN pip install -r requirements.txt

# Start uWSGI using config file
CMD ["uwsgi", "app.ini"]