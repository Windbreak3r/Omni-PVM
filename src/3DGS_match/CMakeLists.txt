cmake_minimum_required(VERSION 3.10)
project(3DGS_match)

set(CMAKE_BUILD_TYPE Debug)
add_compile_options(-g)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# set(G2O_DIR /home/pjm/lib/g2o-master/cmake_modules)
# 手动指定 g2o 的路径（因为你编译安装在 /usr/local）
set(G2O_INCLUDE_DIRS /usr/local/include)
set(G2O_LIBRARIES
    /usr/local/lib/libg2o_core.so
    /usr/local/lib/libg2o_stuff.so
    /usr/local/lib/libg2o_types_slam3d.so
    /usr/local/lib/libg2o_solver_csparse.so
    /usr/local/lib/libg2o_csparse_extension.so
    /usr/local/lib/libg2o_types_sba.so
)

set(VTK_ROOT_DIR "/opt/vtk-9.2.2")
set(VTK_DIR "/opt/vtk-9.2.2/lib/cmake/vtk-9.2")

list(APPEND CMAKE_MODULE_PATH /home/pjm/lib/g2o-master/cmake_modules)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rosbag
  cv_bridge
  sensor_msgs
  std_msgs
  image_transport
)

find_package(OpenCV 4.10 REQUIRED)
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES viterbi
  CATKIN_DEPENDS roscpp std_msgs image_transport cv_bridge sensor_msgs rosbag
  DEPENDS OpenCV
)

# 添加 G2O 库的查找
# find_package(G2O REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem iostreams)
find_package(Eigen3 REQUIRED)

find_package(VTK 9.2 REQUIRED COMPONENTS
  CommonCore
  InteractionStyle
  RenderingCore
  FiltersCore
)
find_package(PCL REQUIRED)

add_executable(3DGS_match
  src/main.cpp
  src/visual_map.cpp
  src/query_information.cpp
  src/HMM_2.cpp
  src/match.cpp
  src/motion_predictor.cpp
  src/pnp_solver.cpp
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

# 为 PCL 和 VTK 设置编译定义
include_directories(${PCL_INCLUDE_DIRS} ${VTK_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})

target_include_directories(3DGS_match PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${G2O_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(3DGS_match
  ${OpenCV_LIBS}
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${Boost_LIBRARIES}
  ${VTK_LIBRARIES}
  ${EIGEN3_LIBS}
  ${G2O_LIBRARIES}
)

target_link_libraries(3DGS_match g2o_core g2o_stuff)
