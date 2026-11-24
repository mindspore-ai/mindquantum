# Copyright 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0.

# lint_cmake: -whitespace/indent

# Keep SDK-created kernels out of development-only installs even when the SDK omits an install component.
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME Runtime)

# Enable ASC once in the common parent of all kernel targets. ASCConfig's changes to host compiler flags stay here,
# outside the host extensions. The SDK accepts one architecture globally; our variants select it per target for both
# compilation and linking instead of passing the full list to bisheng.
set(CMAKE_ASC_ARCHITECTURES "")
find_package(ASC REQUIRED HINTS "${ASCEND_CANN_PACKAGE_PATH}/lib64/cmake")
enable_language(ASC)

function(mq_ascend_kernel_library name host)
  set(target "${name}_${MQ_ASCEND_KERNEL_TARGET}")
  if(MQ_ASCEND_SOC)
    set(selector "--npu-soc=${MQ_ASCEND_KERNEL_TARGET}")
  else()
    set(selector "--npu-arch=${MQ_ASCEND_KERNEL_TARGET}")
  endif()
  set(CMAKE_INSTALL_LIBDIR "lib/ascend")
  if(NOT IS_PYTHON_BUILD)
    set(CMAKE_INSTALL_LIBDIR "${MQ_INSTALL_PYTHONDIR}/${MQ_PYTHON_PACKAGE_NAME}/lib/ascend")
  endif()
  ascendc_library(${target} SHARED ${ARGN})
  target_include_directories(${target} PRIVATE "${PROJECT_SOURCE_DIR}/ccsrc/include" "${MQ_ASCEND_INCLUDE}")
  target_compile_options(
    ${target}
    PRIVATE "${selector}"
            -O3
            -fno-fast-math
            -ffp-contract=off
            -fvisibility=hidden
            "$<$<BOOL:${ENABLE_STACK_PROTECTION}>:SHELL:-Xhost-start -fstack-protector-all -Xhost-end>"
            "$<$<CONFIG:Release>:SHELL:-Xhost-start -D_FORTIFY_SOURCE=2 -Xhost-end>")
  target_compile_definitions(${target} PRIVATE ASCENDC_DUMP=0)
  # The SDK exposes its runtime dependencies to consumers. A dlopen library must link them itself; the host must not
  # acquire a DT_NEEDED on a variant.
  get_target_property(libraries ${target} INTERFACE_LINK_LIBRARIES)
  get_target_property(directories ${target} INTERFACE_LINK_DIRECTORIES)
  target_link_libraries(${target} PRIVATE ${libraries})
  target_link_directories(${target} PRIVATE ${directories})
  target_link_options(
    ${target}
    PRIVATE
    "${selector}"
    -Wl,-z,defs
    "$<$<BOOL:${LINKER_RELRO}>:-Wl,-z,relro>"
    "$<$<BOOL:${LINKER_NOW}>:-Wl,-z,now>"
    "$<$<BOOL:${LINKER_NOEXECSTACK}>:-Wl,-z,noexecstack>"
    "$<$<AND:$<CONFIG:Release>,$<BOOL:${LINKER_STRIP_ALL}>>:-Wl,--strip-all>")
  set_target_properties(${target} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "$<TARGET_FILE_DIR:${host}>/lib/ascend"
                                             DEBUG_POSTFIX "")
  add_dependencies(${host} ${target})
  set_property(GLOBAL APPEND PROPERTY _python_runtime_targets ${target})
  set_property(
    GLOBAL APPEND PROPERTY _python_runtime_files
                           "lib/ascend/${CMAKE_SHARED_LIBRARY_PREFIX}${target}${CMAKE_SHARED_LIBRARY_SUFFIX}")
endfunction()

foreach(MQ_ASCEND_KERNEL_TARGET IN LISTS MQ_ASCEND_TARGETS)
  add_subdirectory("${PROJECT_SOURCE_DIR}/ccsrc/lib/simulator/vector/ascend/kernels"
                   "${MQ_ASCEND_KERNEL_TARGET}/vector")
  add_subdirectory("${PROJECT_SOURCE_DIR}/ccsrc/lib/algorithm/qaia/ascend/kernels" "${MQ_ASCEND_KERNEL_TARGET}/qaia")
endforeach()
