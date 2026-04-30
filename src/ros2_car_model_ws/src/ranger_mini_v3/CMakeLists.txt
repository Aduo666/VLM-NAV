cmake_minimum_required(VERSION 3.5)
project(ranger_mini_v3)

# 1️⃣ 使用 ament_cmake 作为构建系统
find_package(ament_cmake REQUIRED)

# 2️⃣ 安装资源文件（URDF、mesh、config、launch 等）
foreach(dir config launch meshes urdf)
  install(DIRECTORY ${dir}/
    DESTINATION share/${PROJECT_NAME}/${dir})
endforeach()

# 3️⃣ 结束宏
ament_package()

