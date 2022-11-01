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
  /Users/jedzhou/VscodeProjects/mindquantum/ccsrc/lib/simulator/utils.cpp
/Users/jedzhou/VscodeProjects/mindquantum/ccsrc/lib/mq_base/utils.cc
)


# ==============================================================================
# add include_lib
add_library(include_lib INTERFACE)

target_include_directories(
  include_lib INTERFACE ${Python3_INCLUDE_DIRS} ${pybind11_INCLUDE_DIRS}
  /Users/jedzhou/VscodeProjects/mindquantum/ccsrc/include
  /Users/jedzhou/VscodeProjects/mindquantum/ccsrc/lib/mq_base)

target_link_libraries(mindquantum_sim PUBLIC include_lib ${Python3_LIBRARIES})

# ================================================================================
# add mqsim_common
add_library(mqsim_common STATIC)

target_sources(mqsim_common PRIVATE ${PROJECT_SOURCE_DIR}/../lib/simulator/utils.cpp
                                    ${PROJECT_SOURCE_DIR}/../lib/simulator/timer.cpp)

target_link_libraries(mqsim_common PUBLIC mq_base include_lib)

# =================================================================================

add_executable(test_dm main.cpp)

target_include_directories(
  test_dm PRIVATE )

  target_link_libraries(test_dm PUBLIC mindquantum_sim)