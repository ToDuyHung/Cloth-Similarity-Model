FROM nginx

WORKDIR /app/
RUN apt-get update
#RUN apt-get install -y --reinstall systemd
#RUN apt-get install -y nano vim procps
RUN apt-get install -y ffmpeg --fix-missing
RUN apt-get install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget -y --fix-missing
RUN cd /tmp \
    && wget https://www.python.org/ftp/python/3.8.3/Python-3.8.3.tgz \
    && tar -xf Python-3.8.3.tgz && cd Python-3.8.3 \
    && ./configure --enable-optimizations \
    && make altinstall \
    && make install
RUN cp /usr/local/bin/pip3.8 /usr/local/bin/pip
RUN pip install uvicorn \
    && pip install fastapi \
    && pip install requests \
    && pip install python-multipart \
    && pip install asyncio

COPY . .
COPY media_server/nginx.conf /etc/nginx/nginx.conf
COPY media_server/nginx.service /lib/systemd/system

ENTRYPOINT ["sh", "start.sh"]