build_from_dockerfile:
	sudo docker build -t video_reciever .

build:
	sudo docker-compose --env-file .env up -d --build --remove-orphans

stop:
	sudo docker-compose down

start:
	sudo docker-compose --env-file .env up -d

restart:
	sudo docker-compose down
	sudo docker-compose --env-file .env up -d
