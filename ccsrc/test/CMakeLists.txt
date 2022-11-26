cmake_minimum_required(VERSION 3.20)
cmake_policy(VERSION 3.0...3.23)

set(CMAKE_C_COMPILER "/usr/local/bin/clang")
set(CMAKE_CXX_COMPILER "/usr/local/bin/clang++")

project(mindquantum_sim)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS YES)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fopenmp -O3 -ffast-math")




# ==============================================================================
# openmp support
find_package(OpenMP)

# ==============================================================================
# python support
find_package (Python3 COMPONENTS Interpreter Development)

# ==============================================================================
# pybind support
find_package (pybind11)

# include_directories(/Users/jedzhou/VscodeProjects/mindquantum/ccsrc/include)

# ===============================================================================
# add mindquantum_sim
add_library(mindquantum_sim STATIC)

set_target_properties(mindquantum_sim PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

target_sources(
  mindquantum_sim PRIVATE
  /Users/jedzhou/VscodeProjects/mindquantum/ccsrc/lib/simulator/densitymatrix/detail/cpu_densitymatrix_policy.cpp 
  /Users/jedzhou/VscodeProjects/mindquantum/ccsrc/lib/simulator/densitymatrix/detail/cpu_densitymatrix_policy_zlike.cpp 
  /Users/jedzhou/VscodeProjects/mindquantum/ccsrc/lib/simulator/densitymatrix/detail/cpu_densitymatrix_policy_xlike.cpp
  /Users/jedzhou/VscodeProjects/mindquantum/ccsrc/lib/simulator/densitymatrix/detail/cpu_densitymatrix_policy_single_ops.cpp
  /Users/jedzhou/VscodeProjects/mindquantum/ccsrc/lib/simulator/densitymatrix/detail/cpu_densitymatrix_policy_two_ops.cpp 
  /Users/jedzhou/VscodeProjects/mindquantum/ccsrc/lib/simulator/densitymatrix/detail/cpu_densitymatrix_policy_channel.cpp 
  /Users/jedzhou/VscodeProjects/mindquantum/ccsrc/lib/simulator/utils.cpp
/Users/jedzhou/VscodeProjects/mindquantum/ccsrc/lib/mq_base/utils.cc
/Users/jedzhou/VscodeProjects/mindquantum/ccsrc/lib/simulator/densitymatrix/detail/cpu_densitymatrix_policy_gate_expec.cpp
)


# ==============================================================================
# add include_lib
add_library(include_lib INTERFACE)

target_include_directories(
  include_lib INTERFACE ${pybind11_INCLUDE_DIRS}
  /Users/jedzhou/VscodeProjects/mindquantum/ccsrc/include
  # /Users/jedzhou/VscodeProjects/mindquantum/ccsrc/lib/mq_base
  /Users/jedzhou/VscodeProjects/mindquantum/third_party/mindquantum/ccsrc/mq_base
  )

target_link_libraries(mindquantum_sim PUBLIC include_lib)

# ================================================================================
# add mqsim_common
add_library(mqsim_common STATIC)

target_sources(mqsim_common PRIVATE ${PROJECT_SOURCE_DIR}/../lib/simulator/utils.cpp
                                    ${PROJECT_SOURCE_DIR}/../lib/simulator/timer.cpp)

target_link_libraries(mqsim_common PUBLIC mq_base include_lib)




# =================================================================================
# add pybind

pybind11_add_module(_mq_densitymatrix MODULE /Users/jedzhou/VscodeProjects/mindquantum/ccsrc/python/simulator/lib/_mq_densitymatrix.cpp)

target_include_directories(_mq_densitymatrix PRIVATE $<BUILD_INTERFACE:/Users/jedzhou/VscodeProjects/mindquantum/ccsrc/python/simulator/include>)
target_link_libraries(_mq_densitymatrix PUBLIC mindquantum_sim)
set_target_properties(_mq_densitymatrix PROPERTIES POSITION_INDEPENDENT_CODE TRUE )

add_library(bind_lib INTERFACE)
target_include_directories(bind_lib INTERFACE $<BUILD_INTERFACE:/Users/jedzhou/VscodeProjects/mindquantum/ccsrc/python/simulator/include>)
target_link_libraries(_mq_densitymatrix PUBLIC bind_lib include_lib)


# add_library(bind_lib INTERFACE)
# target_include_directories(bind_lib INTERFACE 
# /Users/jedzhou/VscodeProjects/mindquantum/ccsrc/python/simulator/include/python/densitymatrix)
# target_link_libraries(mindquantum_sim PUBLIC bind_lib)

# =================================================================================

add_executable(test_policy test_policy.cpp)

target_link_libraries(test_policy PUBLIC mindquantum_sim)

# =================================================================================

add_executable(test_state test_state.cpp)

target_link_libraries(test_state PUBLIC mindquantum_sim)
