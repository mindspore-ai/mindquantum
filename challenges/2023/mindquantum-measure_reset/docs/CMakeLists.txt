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

find_package(Sphinx REQUIRED)

# ==============================================================================

set(_mindquantum_public_headers)
get_property(_doc_targets GLOBAL PROPERTY _doc_targets)

foreach(tgt ${_doc_targets})
  get_target_property(_header_list ${tgt} INTERFACE_SOURCES)
  if(_header_list)
    list(APPEND _mindquantum_public_headers ${_header_list})
  endif()
endforeach()

# ------------------------------------------------------------------------------

set(_mindquantum_rest_doc_files)
file(GLOB_RECURSE _mindquantum_rest_doc_files "${CMAKE_CURRENT_LIST_DIR}/*.rst" "${CMAKE_CURRENT_LIST_DIR}/*.md")

# ==============================================================================
# Doxygen

set(DOXYGEN_INPUT_DIR ${PROJECT_SOURCE_DIR}/ccsrc)
set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/doxygen)
set(DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIR}/html/index.html)
set(DOXYFILE_IN ${CMAKE_CURRENT_LIST_DIR}/Doxyfile)
set(DOXYFILE_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

# ------------------------------------------------------------------------------

file(READ ${DOXYFILE_IN} _doxyfile_content)
string(REPLACE "INPUT                  = ../ccsrc" "INPUT                  = ${DOXYGEN_INPUT_DIR}" _doxyfile_content
               ${_doxyfile_content})
# Not using DOXYGEN_OUTPUT_DIR because Doxygen adds '/doxygen' to the end of this path
string(REPLACE "OUTPUT_DIRECTORY       =" "OUTPUT_DIRECTORY       = ${CMAKE_CURRENT_BINARY_DIR}" _doxyfile_content
               ${_doxyfile_content})
file(WRITE ${DOXYFILE_OUT} ${_doxyfile_content})
file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR})

# ------------------------------------------------------------------------------

add_custom_command(
  OUTPUT ${DOXYGEN_INDEX_FILE}
  DEPENDS ${_mindquantum_public_headers}
  COMMAND $<TARGET_FILE:Doxygen::doxygen> ${DOXYFILE_OUT}
  MAIN_DEPENDENCY ${DOXYFILE_OUT}
  ${DOXYFILE_IN}
  COMMENT "Generating docs")

add_custom_target(
  doxygen
  DEPENDS ${DOXYGEN_INDEX_FILE}
  COMMENT "Execute doxygen on the project.")

# ==============================================================================
# Sphinx (sphinx-build)

set(SPHINX_SOURCE ${CMAKE_CURRENT_LIST_DIR}/source)
set(SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/html)
set(SPHINX_INDEX_FILE ${SPHINX_BUILD}/index.html)

# ------------------------------------------------------------------------------

# Only regenerate Sphinx when: - Doxygen has rerun - Our doc files have been updated - The Sphinx config has been
# updated

add_custom_command(
  OUTPUT ${SPHINX_INDEX_FILE}
  COMMAND
    $<TARGET_FILE:Python::Interpreter> ${Sphinx_EXECUTABLE} -b html
    # Tell Breathe where to find the Doxygen output
    -Dbreathe_projects.mindquantum=${DOXYGEN_OUTPUT_DIR}/xml ${SPHINX_SOURCE} ${SPHINX_BUILD}
  WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
  DEPENDS # Other docs files you want to track should go here (or in some variable)
          ${_mindquantum_rest_doc_files} ${DOXYGEN_INDEX_FILE}
  MAIN_DEPENDENCY ${SPHINX_SOURCE}/conf.py
  COMMENT "Generating documentation with Sphinx")

# ==============================================================================
# Sphinx (sphinx-build)

set(DOCS_INDEX_FILE ${CMAKE_CURRENT_BINARY_DIR}/index.html)
set(DOCS_INDEX_FILE_IN ${CMAKE_CURRENT_LIST_DIR}/index.html.in)

# ------------------------------------------------------------------------------

add_custom_command(
  OUTPUT ${DOCS_INDEX_FILE}
  COMMAND ${CMAKE_COMMAND} -E copy ${DOCS_INDEX_FILE_IN} ${DOCS_INDEX_FILE}
  COMMENT "Generate index.html file in build directory")

# ==============================================================================
# Add an install target to install the docs

# Nice named target so we can run the job easily
add_custom_target(
  docs
  DEPENDS doxygen ${SPHINX_INDEX_FILE} ${DOCS_INDEX_FILE}
  COMMENT "Generate documentation")

# ==============================================================================

# NB: the SPHINX_BUILD directory will get populated by sphinx-build if (ever) called
file(MAKE_DIRECTORY ${SPHINX_BUILD})
install(DIRECTORY ${SPHINX_BUILD} DESTINATION ${MQ_INSTALL_DOCDIR})
