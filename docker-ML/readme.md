# 1. X11 접근 허용
xhost +local:docker

# 2. docker-ML 경로로 이동 후 실행
cd ~/LearningByCheating_ws/docker-ML
sudo docker compose -f docker-compse.pc.yaml build --no-cache
sudo docker compose -f docker-compse.pc.yaml up -d --force-recreate

# 3. 컨테이너 접속
docker exec -it ml-noetic-pc bash

# 4. (컨테이너 안)
source /opt/ros/noetic/setup.bash
roscore