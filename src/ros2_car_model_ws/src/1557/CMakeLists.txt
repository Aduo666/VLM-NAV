cmake_minimum_required(VERSION 3.5)
project(1557)

find_package(ament_cmake REQUIRED)

# 安装资源文件（URDF、launch、rviz、config、meshes）
foreach(dir launch urdf rviz config meshes)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${dir})
    install(DIRECTORY ${dir}/
      DESTINATION share/${PROJECT_NAME}/${dir})
  endif()
endforeach()

ament_package()

