# ==============================================================================
#
# Copyright 2022 <Huawei Technologies Co., Ltd>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================

# lint_cmake: -whitespace/indent

pybind11_add_module(_mq_vector MODULE ${CMAKE_CURRENT_SOURCE_DIR}/lib/_mq_vector.cpp
                    OUTPUT_HINT "${MQ_PYTHON_PACKAGE_NAME}")

target_include_directories(_mq_vector PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>)
force_at_least_cxx17_workaround(_mq_vector)
target_link_libraries(_mq_vector PUBLIC mq_python_core mqsim_vector_cpu)

# ------------------------------------------------------------------------------

if(ENABLE_CUDA)
  pybind11_add_module(_mq_vector_gpu MODULE ${CMAKE_CURRENT_SOURCE_DIR}/lib/_mq_vector_gpu.cu
                      OUTPUT_HINT "${MQ_PYTHON_PACKAGE_NAME}")

  target_include_directories(_mq_vector_gpu PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>)
  force_at_least_cxx17_workaround(_mq_vector_gpu)
  target_link_libraries(_mq_vector_gpu PUBLIC mq_python_core mqsim_vector_gpu)
  set_target_properties(_mq_vector_gpu PROPERTIES CUDA_RUNTIME_LIBRARY $<IF:$<BOOL:${CUDA_STATIC}>,Static,Shared>)
endif()

# ------------------------------------------------------------------------------

find_package(ASCENDCL QUIET)
if(ASCENDCL_FOUND)
  set(MQVECTOR_NPU_KERNEL_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/lib/mqvector_npu_kernel.cpp)

  # Use CANN's complete Ascend C build chain. In addition to compiling the
  # device code, this generates the host stub and packs the device binary into
  # the host object consumed by the extension.
  get_filename_component(_ascendcl_lib_dir "${ASCENDCL_LIBRARY}" DIRECTORY)
  get_filename_component(_ascend_cann_root "${_ascendcl_lib_dir}" DIRECTORY)
  set(ASCEND_CANN_PACKAGE_PATH "${_ascend_cann_root}")
  set(ASCEND_PYTHON_EXECUTABLE "${Python_EXECUTABLE}")
  set(MQVECTOR_NPU_SOC_VERSION
      "ascend910b2c"
      CACHE STRING "Ascend SoC version used to compile mqvector_npu kernels")
  if(NOT ASCENDCL_DEVKIT_ROOT)
    message(FATAL_ERROR "ASCENDCL_DEVKIT_ROOT was not detected by FindASCENDCL.")
  endif()
  set(_ascendc_cmake "${ASCENDCL_DEVKIT_ROOT}/tikcpp/ascendc_kernel_cmake/ascendc.cmake")
  if(NOT EXISTS "${_ascendc_cmake}")
    message(
      FATAL_ERROR
        "Unable to locate the Ascend C CMake package at ${_ascendc_cmake}. "
        "Check the CANN installation and target architecture.")
  endif()
  set(SOC_VERSION "${MQVECTOR_NPU_SOC_VERSION}")
  set(RUN_MODE "npu")
  set(ASCEND_KERNEL_LAUNCH_ONLY OFF)
  include("${_ascendc_cmake}")
  ascendc_library(
    mqvector_npu_kernel_ascendc
    STATIC
    ${MQVECTOR_NPU_KERNEL_SOURCE})

  pybind11_add_module(_mq_vector_npu MODULE ${CMAKE_CURRENT_SOURCE_DIR}/lib/_mq_vector_npu.cpp
                      OUTPUT_HINT "${MQ_PYTHON_PACKAGE_NAME}")

  target_include_directories(_mq_vector_npu PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>)
  force_at_least_cxx17_workaround(_mq_vector_npu)
  target_link_libraries(_mq_vector_npu PUBLIC mq_python_core mq_math ASCENDCL::ascendcl
                                              PRIVATE mqvector_npu_kernel_ascendc)
  add_dependencies(_mq_vector_npu mqvector_npu_kernel_ascendc)
else()
  message(STATUS "CANN/AscendCL was not found; skipping the optional mqvector_npu extension.")
endif()

# ------------------------------------------------------------------------------

# lint_cmake: -whitespace/indent

pybind11_add_module(_mq_mps MODULE ${CMAKE_CURRENT_SOURCE_DIR}/lib/_mq_mps.cpp OUTPUT_HINT "${MQ_PYTHON_PACKAGE_NAME}")

target_include_directories(_mq_mps PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>)
force_at_least_cxx17_workaround(_mq_mps)
target_link_libraries(_mq_mps PUBLIC mq_python_core mqsim_mps_cpu)

# ------------------------------------------------------------------------------

# lint_cmake: -whitespace/indent

pybind11_add_module(_mq_matrix MODULE ${CMAKE_CURRENT_SOURCE_DIR}/lib/_mq_matrix.cpp
                    OUTPUT_HINT "${MQ_PYTHON_PACKAGE_NAME}")

target_include_directories(_mq_matrix PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>)
force_at_least_cxx17_workaround(_mq_matrix)
target_link_libraries(_mq_matrix PUBLIC mq_python_core mqsim_densitymatrix_cpu)

# ------------------------------------------------------------------------------

pybind11_add_module(_mq_stabilizer MODULE ${CMAKE_CURRENT_SOURCE_DIR}/lib/_mq_stabilizer.cpp
                    OUTPUT_HINT "${MQ_PYTHON_PACKAGE_NAME}")

target_include_directories(_mq_stabilizer PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>)
force_at_least_cxx17_workaround(_mq_stabilizer)
target_link_libraries(_mq_stabilizer PUBLIC mq_python_core mqsim_stabilizer)

pybind11_add_module(_mq_chem MODULE ${CMAKE_CURRENT_SOURCE_DIR}/lib/_mq_chem.cpp
                    OUTPUT_HINT "${MQ_PYTHON_PACKAGE_NAME}")

target_include_directories(_mq_chem PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>)
force_at_least_cxx17_workaround(_mq_chem)
target_link_libraries(_mq_chem PUBLIC mq_python_core mqchem_cpu)

# ------------------------------------------------------------------------------

add_subdirectory(include)
