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

# lint_cmake: -whitespace/indent, -convention/filename,-package/stdargs

find_program(M4_EXEC m4 PATHS /usr/local/bin /usr/bin /bin /sbin)

if(NOT M4_EXEC AND m4_FIND_REQUIRED)
  message(
    SEND_ERROR "Unable to locate m4 executable. "
               "Try to install it and re-run CMake:\n"
               "  - Ubuntu/Debian: sudo apt-get install m4\n"
               "  - ArchLinux: sudo pacman -S m4\n"
               "  - Fedora/CentOS: sudo yum install m4 or sudo dnf install m4\n"
               "  - MacOS Homebrew: brew install m4\n"
               "  - MacOS MacPorts: sudo port install m4")
endif()

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(m4 REQUIRED_VARS M4_EXEC)

mark_as_advanced(M4_EXEC)
