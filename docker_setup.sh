#!/bin/bash

PATH_TO_PROJECT="$(dirname "$(realpath "$0")")"
echo "PATH_TO_PROJECT: $PATH_TO_PROJECT"

# replace "/abs/path/to/ACDC" with the absolute path to the ACDC directory
python $PATH_TO_PROJECT/data/os_interaction/images.py build -c $PATH_TO_PROJECT/data/os_interaction/configs/std.yaml -r .
docker run -d -p 5800:5672 -v $PATH_TO_PROJECT/utils/rabbitmq.conf:/etc/rabbitmq/rabbitmq.conf rabbitmq
docker run -d -p 6500:6379 -v $PATH_TO_PROJECT/utils/redis.conf:/etc/redis/redis.conf redis redis-server /etc/redis/redis.conf
docker run -d -p 5801:5672 -v $PATH_TO_PROJECT/utils/rabbitmq.conf:/etc/rabbitmq/rabbitmq.conf rabbitmq
docker run -d -p 6501:6379 -v $PATH_TO_PROJECT/utils/redis.conf:/etc/redis/redis.conf redis redis-server /etc/redis/redis.conf
docker run -d -p 5802:5672 -v $PATH_TO_PROJECT/utils/rabbitmq.conf:/etc/rabbitmq/rabbitmq.conf rabbitmq
docker run -d -p 6502:6379 -v $PATH_TO_PROJECT/utils/redis.conf:/etc/redis/redis.conf redis redis-server /etc/redis/redis.conf
docker run -d -p 5803:5672 -v $PATH_TO_PROJECT/utils/rabbitmq.conf:/etc/rabbitmq/rabbitmq.conf rabbitmq
docker run -d -p 6503:6379 -v $PATH_TO_PROJECT/utils/redis.conf:/etc/redis/redis.conf redis redis-server /etc/redis/redis.conf
docker run -d -p 5804:5672 -v $PATH_TO_PROJECT/utils/rabbitmq.conf:/etc/rabbitmq/rabbitmq.conf rabbitmq
docker run -d -p 6504:6379 -v $PATH_TO_PROJECT/utils/redis.conf:/etc/redis/redis.conf redis redis-server /etc/redis/redis.conf
docker run -d -p 5805:5672 -v $PATH_TO_PROJECT/utils/rabbitmq.conf:/etc/rabbitmq/rabbitmq.conf rabbitmq
docker run -d -p 6505:6379 -v $PATH_TO_PROJECT/utils/redis.conf:/etc/redis/redis.conf redis redis-server /etc/redis/redis.conf
docker run -d -p 5806:5672 -v $PATH_TO_PROJECT/utils/rabbitmq.conf:/etc/rabbitmq/rabbitmq.conf rabbitmq
docker run -d -p 6506:6379 -v $PATH_TO_PROJECT/utils/redis.conf:/etc/redis/redis.conf redis redis-server /etc/redis/redis.conf
docker run -d -p 5807:5672 -v $PATH_TO_PROJECT/utils/rabbitmq.conf:/etc/rabbitmq/rabbitmq.conf rabbitmq
docker run -d -p 6507:6379 -v $PATH_TO_PROJECT/utils/redis.conf:/etc/redis/redis.conf redis redis-server /etc/redis/redis.conf
