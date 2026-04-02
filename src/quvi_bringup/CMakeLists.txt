cmake_minimum_required(VERSION 3.8)
project(quvi_bringup)

find_package(ament_cmake REQUIRED)

# Launch 파일 설치
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# 설정 파일 설치
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

ament_package()
