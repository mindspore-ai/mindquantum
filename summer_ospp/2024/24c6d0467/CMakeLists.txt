cmake_minimum_required(VERSION 3.0)
project(example_person)

# 如果代码需要支持C++11，就直接加上这句
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
# 如果想要生成的可执行文件拥有符号表，可以gdb调试，就直接加上这句
add_definitions("-Wall -g")

# 设置变量，下面的代码都可以用到
set(SIMD_DIR ${PROJECT_SOURCE_DIR}/simd)
set(CIRCUIT_DIR ${PROJECT_SOURCE_DIR}/circuit)
set(ARG_DIR ${PROJECT_SOURCE_DIR}/arg)
set(TABLEAU_DIR ${PROJECT_SOURCE_DIR}/tableau)
# 编译子文件夹的CMakeLists.txt
add_subdirectory(proto_pb2)

# 规定.h头文件路径
include_directories(${PROJECT_SOURCE_DIR}
    ${TABLEAU_DIR} ${SIMD_DIR} ${CIRCUIT_DIR} ${ARG_DIR}
)

# 生成可执行文件
add_executable(${PROJECT_NAME} stabilizer_tableau_main.cpp )
