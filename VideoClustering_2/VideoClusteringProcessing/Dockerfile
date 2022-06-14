FROM python:3.8-slim-buster

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install vim nano -y

RUN pip3 install sklearn
RUN pip3 install imutils
RUN pip3 install tensorflow==2.6.0rc0
RUN pip3 install keras==2.6.*
RUN pip3 install scikit-learn
RUN pip3 install numpy>=1.18.5
RUN pip3 install opencv-python>=4.1.2
RUN pip3 install Pillow>=7.1.2
RUN pip3 install PyYAML>=5.3.1
RUN pip3 install requests>=2.23.0
RUN pip3 install scipy==1.8.0
RUN pip3 install tqdm>=4.41.0
RUN pip3 install tensorboard>=2.4.1
RUN pip3 install pandas>=1.1.4
RUN pip3 install seaborn>=0.11.0
RUN pip3 install torch==1.9.1
RUN pip3 install torchvision==0.10.1
RUN pip3 install protobuf==3.20.1
RUN pip3 install memory_profiler
RUN pip3 install webcolors
RUN pip3 install kafka-python

COPY . .

RUN export PYTHONPATH=./
CMD ["python", "app.py"]