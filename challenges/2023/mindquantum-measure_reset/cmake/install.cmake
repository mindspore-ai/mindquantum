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

include(CMakePackageConfigHelpers)

set(_namespace mindquantum::)

get_property(mq_install_targets GLOBAL PROPERTY mq_install_targets)
list(REMOVE_DUPLICATES mq_install_targets)

# ==============================================================================

set(MQ_INSTALL_IN_BUILD_DIR TRUE)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/mindquantumConfig.cmake.in ${PROJECT_BINARY_DIR}/mindquantumConfig.cmake
  INSTALL_DESTINATION ${PROJECT_BINARY_DIR}
  INSTALL_PREFIX ${PROJECT_BINARY_DIR})

# --------------------------------------

set(MQ_INSTALL_IN_BUILD_DIR FALSE)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/mindquantumConfig.cmake.in ${PROJECT_BINARY_DIR}/config_for_install/mindquantumConfig.cmake
  INSTALL_DESTINATION ${MQ_INSTALL_CMAKEDIR})

write_basic_package_version_file(
  ${PROJECT_BINARY_DIR}/mindquantumConfigVersion.cmake
  COMPATIBILITY SameMajorVersion
  VERSION ${MQ_VERSION})

install(FILES ${PROJECT_BINARY_DIR}/config_for_install/mindquantumConfig.cmake
              ${PROJECT_BINARY_DIR}/mindquantumConfigVersion.cmake DESTINATION ${MQ_INSTALL_CMAKEDIR})

# ==============================================================================

install(DIRECTORY ${PROJECT_SOURCE_DIR}/cmake/Modules ${PROJECT_SOURCE_DIR}/cmake/commands
        DESTINATION ${MQ_INSTALL_CMAKEDIR})
install(FILES "${CMAKE_CURRENT_LIST_DIR}/packages.cmake" DESTINATION ${MQ_INSTALL_CMAKEDIR})

install(DIRECTORY ${PROJECT_SOURCE_DIR}/cmake/NVCXX DESTINATION ${MQ_INSTALL_CMAKEDIR}/Modules)

# ------------------------------------------------------------------------------

install(
  TARGETS ${mq_install_targets}
  EXPORT mindquantumTargets
  PRIVATE_HEADER DESTINATION ${MQ_INSTALL_INCLUDEDIR}
  PUBLIC_HEADER DESTINATION ${MQ_INSTALL_INCLUDEDIR}
  ARCHIVE DESTINATION ${MQ_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${MQ_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${MQ_INSTALL_BINDIR})

install(
  EXPORT mindquantumTargets
  NAMESPACE ${_namespace}
  DESTINATION ${MQ_INSTALL_CMAKEDIR})

# NB: if called from setup.py, we do not need to care about installing the Python related targets, as this will be taken
# care of by Python directly.
if(NOT IS_PYTHON_BUILD)
  install(
    EXPORT mindquantumPythonTargets
    NAMESPACE ${_namespace}
    DESTINATION ${MQ_INSTALL_CMAKEDIR})

  export(
    EXPORT mindquantumPythonTargets
    NAMESPACE ${_namespace}
    FILE mindquantumPythonTargets.cmake)
endif()

# ==============================================================================

export(
  EXPORT mindquantumTargets
  NAMESPACE ${_namespace}
  FILE mindquantumTargets.cmake)
export(PACKAGE mindquantum)
