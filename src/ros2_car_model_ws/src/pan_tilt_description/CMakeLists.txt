cmake_minimum_required(VERSION 3.5)
project(pan_tilt_description)

# 1️⃣ 使用 ament_cmake 作为构建系统
find_package(ament_cmake REQUIRED)
find_package(urdf REQUIRED)

# 2️⃣ 安装资源文件
install(DIRECTORY urdf launch meshes
  DESTINATION share/${PROJECT_NAME}
)

# 3️⃣ 结束宏
ament_package()

