# Copyright 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0.

# CMake owns feature selection. Packaging must distinguish an absent target from a failed build.
get_property(_python_targets GLOBAL PROPERTY _python_targets)
list(REMOVE_DUPLICATES _python_targets)
list(JOIN _python_targets "\", \"" _python_targets_json)
if(_python_targets)
  set(_python_targets_json "\"${_python_targets_json}\"")
endif()
set(_ascend_info "null")
if(ENABLE_ASCEND)
  if(CMAKE_ASC_ARCHITECTURES)
    set(_ascend_info "{\"architecture\": \"${CMAKE_ASC_ARCHITECTURES}\"}")
  else()
    set(_ascend_info "{\"soc\": \"${MQ_ASCEND_SOC}\"}")
  endif()
endif()
file(
  CONFIGURE
  OUTPUT "${PROJECT_BINARY_DIR}/python-build-info.json"
  CONTENT
    [[
{
  "python_targets": [@_python_targets_json@],
  "ascend": @_ascend_info@
}
]]
  @ONLY)
