cmake_minimum_required(VERSION 3.13)
project(example_person)
# 如果代码需要支持C++11，就直接加上这句
set(CMAKE_CXX_STANDARD 20)
# 如果想要生成的可执行文件拥有符号表，可以gdb调试，就直接加上这句
# add_definitions("-Wall -g")

set(MACHINE_FLAG "-mavx2" "-msse2")
set(CMAKE_VERBOSE_MAKEFILEON ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY out)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY out)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY out)

# 设置变量，下面的代码都可以用到
file(STRINGS file/key_file.txt KEY_FILE)
file(STRINGS file/test_file.txt TEST_FILE)
# 编译子文件夹的CMakeLists.txt

# 规定.h头文件路径
include_directories(${PROJECT_SOURCE_DIR})
# message("${PROJECT_SOURE_DIR}")
# message("${TEST_DIR}")
# 生成可执行文件
# add_executable(${PROJECT_NAME} stabilizer_tableau_main.cpp)
find_package(GTest QUIET)
if(${GTest_FOUND})
    add_executable(stab_test ${KEY_FILE} ${TEST_FILE})
    target_link_libraries(stab_test GTest::gtest GTest::gtest_main)
    target_compile_options(stab_test PRIVATE -Wall -Wpedantic -g -fno-omit-frame-pointer -fno-strict-aliasing -fsanitize=undefined -fsanitize=address ${MACHINE_FLAG})
    target_link_options(stab_test PRIVATE -g -fno-omit-frame-pointer -fsanitize=undefined -fsanitize=address)
else()
    message("WARNING: Skipped stim_test target. `GTest` not found. To fix, follow Standalone CMake Project install instructions at https://github.com/google/googletest/blob/master/googletest/README.md")
endif()
