# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

# This file sets the basic flags for the C++ language in CMake. It also loads the available platform file for the
# system-compiler if it exists. It also loads a system - compiler - processor (or target hardware) specific file, which
# is mainly useful for crosscompiling and embedded systems.

include(CMakeLanguageInformation)

# some compilers use different extensions (e.g. sdcc uses .rel) so set the extension here first so it can be overridden
# by the compiler specific file
if(UNIX)
  set(CMAKE_NVCXX_OUTPUT_EXTENSION .o)
else()
  set(CMAKE_NVCXX_OUTPUT_EXTENSION .obj)
endif()

set(INCLUDED_FILE 0)

# Load compiler-specific information.
if(CMAKE_NVCXX_COMPILER_ID)
  include(${CMAKE_NVCXX_COMPILER_ID}-NVCXX OPTIONAL)
endif()

set(CMAKE_BASE_NAME)
get_filename_component(CMAKE_BASE_NAME "${CMAKE_CXX_COMPILER}" NAME_WE)

# load a hardware specific file, mostly useful for embedded compilers
if(CMAKE_SYSTEM_PROCESSOR)
  if(CMAKE_NVCXX_COMPILER_ID)
    include(Platform/${CMAKE_EFFECTIVE_SYSTEM_NAME}-${CMAKE_NVCXX_COMPILER_ID}-CXX-${CMAKE_SYSTEM_PROCESSOR} OPTIONAL
            RESULT_VARIABLE INCLUDED_FILE)
  endif()
  if(NOT INCLUDED_FILE)
    include(Platform/${CMAKE_EFFECTIVE_SYSTEM_NAME}-${CMAKE_BASE_NAME}-${CMAKE_SYSTEM_PROCESSOR} OPTIONAL)
  endif()
endif()

# load the system- and compiler specific files
if(CMAKE_NVCXX_COMPILER_ID)
  include(Platform/${CMAKE_EFFECTIVE_SYSTEM_NAME}-${CMAKE_NVCXX_COMPILER_ID}-CXX OPTIONAL RESULT_VARIABLE INCLUDED_FILE)
endif()
if(NOT INCLUDED_FILE)
  include(Platform/${CMAKE_EFFECTIVE_SYSTEM_NAME}-${CMAKE_BASE_NAME} OPTIONAL RESULT_VARIABLE INCLUDED_FILE)
endif()

# load any compiler-wrapper specific information
if(CMAKE_NVCXX_COMPILER_WRAPPER)
  __cmake_include_compiler_wrapper(NVCXX)
endif()

# We specify the compiler information in the system file for some platforms, but this language may not have been enabled
# when the file was first included.  Include it again to get the language info. Remove this when all compiler info is
# removed from system files.
if(NOT INCLUDED_FILE)
  include(Platform/${CMAKE_SYSTEM_NAME} OPTIONAL)
endif()

if(CMAKE_NVCXX_SIZEOF_DATA_PTR)
  foreach(_file ${CMAKE_NVCXX_ABI_FILES})
    include(${_file})
  endforeach()
  unset(CMAKE_NVCXX_ABI_FILES)
endif()

# This should be included before the _INIT variables are used to initialize the cache.  Since the rule variables have if
# blocks on them, users can still define them here. But, it should still be after the platform file so changes can be
# made to those values.

if(CMAKE_USER_MAKE_RULES_OVERRIDE)
  # Save the full path of the file so try_compile can use it.
  include(${CMAKE_USER_MAKE_RULES_OVERRIDE} RESULT_VARIABLE _override)
  set(CMAKE_USER_MAKE_RULES_OVERRIDE "${_override}")
endif()

if(CMAKE_USER_MAKE_RULES_OVERRIDE_NVCXX)
  # Save the full path of the file so try_compile can use it.
  include(${CMAKE_USER_MAKE_RULES_OVERRIDE_NVCXX} RESULT_VARIABLE _override)
  set(CMAKE_USER_MAKE_RULES_OVERRIDE_NVCXX "${_override}")
endif()

# Create a set of shared library variable specific to C++ For 90% of the systems, these are the same flags as the C
# versions so if these are not set just copy the flags from the c version
if(NOT CMAKE_SHARED_LIBRARY_CREATE_NVCXX_FLAGS)
  set(CMAKE_SHARED_LIBRARY_CREATE_NVCXX_FLAGS ${CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS})
endif()

if(NOT CMAKE_NVCXX_COMPILE_OPTIONS_PIC)
  set(CMAKE_NVCXX_COMPILE_OPTIONS_PIC ${CMAKE_C_COMPILE_OPTIONS_PIC})
endif()

if(NOT CMAKE_NVCXX_COMPILE_OPTIONS_PIE)
  set(CMAKE_NVCXX_COMPILE_OPTIONS_PIE ${CMAKE_C_COMPILE_OPTIONS_PIE})
endif()
if(NOT CMAKE_NVCXX_LINK_OPTIONS_PIE)
  set(CMAKE_NVCXX_LINK_OPTIONS_PIE ${CMAKE_C_LINK_OPTIONS_PIE})
endif()
if(NOT CMAKE_NVCXX_LINK_OPTIONS_NO_PIE)
  set(CMAKE_NVCXX_LINK_OPTIONS_NO_PIE ${CMAKE_C_LINK_OPTIONS_NO_PIE})
endif()

if(NOT CMAKE_NVCXX_COMPILE_OPTIONS_DLL)
  set(CMAKE_NVCXX_COMPILE_OPTIONS_DLL ${CMAKE_C_COMPILE_OPTIONS_DLL})
endif()

if(NOT CMAKE_SHARED_LIBRARY_NVCXX_FLAGS)
  set(CMAKE_SHARED_LIBRARY_NVCXX_FLAGS ${CMAKE_SHARED_LIBRARY_C_FLAGS})
endif()

if(NOT DEFINED CMAKE_SHARED_LIBRARY_LINK_NVCXX_FLAGS)
  set(CMAKE_SHARED_LIBRARY_LINK_NVCXX_FLAGS ${CMAKE_SHARED_LIBRARY_LINK_C_FLAGS})
endif()

if(NOT CMAKE_SHARED_LIBRARY_RUNTIME_NVCXX_FLAG)
  set(CMAKE_SHARED_LIBRARY_RUNTIME_NVCXX_FLAG ${CMAKE_SHARED_LIBRARY_RUNTIME_C_FLAG})
endif()

if(NOT CMAKE_SHARED_LIBRARY_RUNTIME_NVCXX_FLAG_SEP)
  set(CMAKE_SHARED_LIBRARY_RUNTIME_NVCXX_FLAG_SEP ${CMAKE_SHARED_LIBRARY_RUNTIME_C_FLAG_SEP})
endif()

if(NOT CMAKE_SHARED_LIBRARY_RPATH_LINK_NVCXX_FLAG)
  set(CMAKE_SHARED_LIBRARY_RPATH_LINK_NVCXX_FLAG ${CMAKE_SHARED_LIBRARY_RPATH_LINK_C_FLAG})
endif()

if(NOT DEFINED CMAKE_EXE_EXPORTS_NVCXX_FLAG)
  set(CMAKE_EXE_EXPORTS_NVCXX_FLAG ${CMAKE_EXE_EXPORTS_C_FLAG})
endif()

if(NOT DEFINED CMAKE_SHARED_LIBRARY_SONAME_NVCXX_FLAG)
  set(CMAKE_SHARED_LIBRARY_SONAME_NVCXX_FLAG ${CMAKE_SHARED_LIBRARY_SONAME_C_FLAG})
endif()

if(NOT CMAKE_EXECUTABLE_RUNTIME_NVCXX_FLAG)
  set(CMAKE_EXECUTABLE_RUNTIME_NVCXX_FLAG ${CMAKE_SHARED_LIBRARY_RUNTIME_NVCXX_FLAG})
endif()

if(NOT CMAKE_EXECUTABLE_RUNTIME_NVCXX_FLAG_SEP)
  set(CMAKE_EXECUTABLE_RUNTIME_NVCXX_FLAG_SEP ${CMAKE_SHARED_LIBRARY_RUNTIME_NVCXX_FLAG_SEP})
endif()

if(NOT CMAKE_EXECUTABLE_RPATH_LINK_NVCXX_FLAG)
  set(CMAKE_EXECUTABLE_RPATH_LINK_NVCXX_FLAG ${CMAKE_SHARED_LIBRARY_RPATH_LINK_NVCXX_FLAG})
endif()

if(NOT DEFINED CMAKE_SHARED_LIBRARY_LINK_NVCXX_WITH_RUNTIME_PATH)
  set(CMAKE_SHARED_LIBRARY_LINK_NVCXX_WITH_RUNTIME_PATH ${CMAKE_SHARED_LIBRARY_LINK_C_WITH_RUNTIME_PATH})
endif()

if(NOT CMAKE_INCLUDE_FLAG_NVCXX)
  set(CMAKE_INCLUDE_FLAG_NVCXX ${CMAKE_INCLUDE_FLAG_C})
endif()

# for most systems a module is the same as a shared library so unless the variable CMAKE_MODULE_EXISTS is set just copy
# the values from the LIBRARY variables
if(NOT CMAKE_MODULE_EXISTS)
  set(CMAKE_SHARED_MODULE_NVCXX_FLAGS ${CMAKE_SHARED_LIBRARY_NVCXX_FLAGS})
  set(CMAKE_SHARED_MODULE_CREATE_NVCXX_FLAGS ${CMAKE_SHARED_LIBRARY_CREATE_NVCXX_FLAGS})
endif()

# repeat for modules
if(NOT CMAKE_SHARED_MODULE_CREATE_NVCXX_FLAGS)
  set(CMAKE_SHARED_MODULE_CREATE_NVCXX_FLAGS ${CMAKE_SHARED_MODULE_CREATE_C_FLAGS})
endif()

if(NOT CMAKE_SHARED_MODULE_NVCXX_FLAGS)
  set(CMAKE_SHARED_MODULE_NVCXX_FLAGS ${CMAKE_SHARED_MODULE_C_FLAGS})
endif()

# Initialize CXX link type selection flags from C versions.
foreach(type SHARED_LIBRARY SHARED_MODULE EXE)
  if(NOT CMAKE_${type}_LINK_STATIC_NVCXX_FLAGS)
    set(CMAKE_${type}_LINK_STATIC_NVCXX_FLAGS ${CMAKE_${type}_LINK_STATIC_C_FLAGS})
  endif()
  if(NOT CMAKE_${type}_LINK_DYNAMIC_NVCXX_FLAGS)
    set(CMAKE_${type}_LINK_DYNAMIC_NVCXX_FLAGS ${CMAKE_${type}_LINK_DYNAMIC_C_FLAGS})
  endif()
endforeach()

if(CMAKE_EXECUTABLE_FORMAT STREQUAL "ELF")
  if(NOT DEFINED CMAKE_NVCXX_LINK_WHAT_YOU_USE_FLAG)
    set(CMAKE_NVCXX_LINK_WHAT_YOU_USE_FLAG "LINKER:--no-as-needed")
  endif()
  if(NOT DEFINED CMAKE_LINK_WHAT_YOU_USE_CHECK)
    set(CMAKE_LINK_WHAT_YOU_USE_CHECK ldd -u -r)
  endif()
endif()

# add the flags to the cache based on the initial values computed in the platform/*.cmake files use _INIT variables so
# that this only happens the first time and you can set these flags in the cmake cache
set(CMAKE_NVCXX_FLAGS_INIT "$ENV{NVCXXFLAGS} ${CMAKE_NVCXX_FLAGS_INIT}")

cmake_initialize_per_config_variable(CMAKE_NVCXX_FLAGS "Flags used by the CXX compiler")

if(CMAKE_NVCXX_STANDARD_LIBRARIES_INIT)
  set(CMAKE_NVCXX_STANDARD_LIBRARIES
      "${CMAKE_NVCXX_STANDARD_LIBRARIES_INIT}"
      CACHE STRING "Libraries linked by default with all NVC++ applications.")
  mark_as_advanced(CMAKE_NVCXX_STANDARD_LIBRARIES)
endif()

# cmake-lint: disable=W0106
if(NOT CMAKE_NVCXX_COMPILER_LAUNCHER AND DEFINED ENV{CMAKE_NVCXX_COMPILER_LAUNCHER})
  set(CMAKE_NVCXX_COMPILER_LAUNCHER
      "$ENV{CMAKE_NVCXX_COMPILER_LAUNCHER}"
      CACHE STRING "Compiler launcher for NVCXX.")
endif()

if(NOT CMAKE_NVCXX_LINKER_LAUNCHER AND DEFINED ENV{CMAKE_NVCXX_LINKER_LAUNCHER})
  set(CMAKE_NVCXX_LINKER_LAUNCHER
      "$ENV{CMAKE_NVCXX_LINKER_LAUNCHER}"
      CACHE STRING "Linker launcher for NVCXX.")
endif()

include(CMakeCommonLanguageInclude)

# now define the following rules: CMAKE_NVCXX_CREATE_SHARED_LIBRARY CMAKE_NVCXX_CREATE_SHARED_MODULE
# CMAKE_NVCXX_COMPILE_OBJECT CMAKE_NVCXX_LINK_EXECUTABLE

# variables supplied by the generator at use time <TARGET> <TARGET_BASE> the target without the suffix <OBJECTS>
# <OBJECT> <LINK_LIBRARIES> <FLAGS> <LINK_FLAGS>

# CXX compiler information <CMAKE_NVCXX_COMPILER> <CMAKE_SHARED_LIBRARY_CREATE_NVCXX_FLAGS>
# <CMAKE_NVCXX_SHARED_MODULE_CREATE_FLAGS> <CMAKE_NVCXX_LINK_FLAGS>

# Static library tools <CMAKE_AR> <CMAKE_RANLIB>

# create a shared C++ library
if(NOT CMAKE_NVCXX_CREATE_SHARED_LIBRARY)
  set(CMAKE_NVCXX_CREATE_SHARED_LIBRARY
      "<CMAKE_NVCXX_COMPILER> <CMAKE_SHARED_LIBRARY_NVCXX_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS>
<CMAKE_SHARED_LIBRARY_CREATE_NVCXX_FLAGS> <SONAME_FLAG><TARGET_SONAME> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")
endif()

# create a c++ shared module copy the shared library rule by default
if(NOT CMAKE_NVCXX_CREATE_SHARED_MODULE)
  set(CMAKE_NVCXX_CREATE_SHARED_MODULE ${CMAKE_NVCXX_CREATE_SHARED_LIBRARY})
endif()

# Create a static archive incrementally for large object file counts. If CMAKE_NVCXX_CREATE_STATIC_LIBRARY is set it
# will override these.
if(NOT DEFINED CMAKE_NVCXX_ARCHIVE_CREATE)
  set(CMAKE_NVCXX_ARCHIVE_CREATE "<CMAKE_AR> qc <TARGET> <LINK_FLAGS> <OBJECTS>")
endif()
if(NOT DEFINED CMAKE_NVCXX_ARCHIVE_APPEND)
  set(CMAKE_NVCXX_ARCHIVE_APPEND "<CMAKE_AR> q <TARGET> <LINK_FLAGS> <OBJECTS>")
endif()
if(NOT DEFINED CMAKE_NVCXX_ARCHIVE_FINISH)
  set(CMAKE_NVCXX_ARCHIVE_FINISH "<CMAKE_RANLIB> <TARGET>")
endif()

# compile a C++ file into an object file
if(NOT CMAKE_NVCXX_COMPILE_OBJECT)
  # Added -x c++ to ensure that *.nvcpp, *.nvcxx files compile properly. This is required for some try_compile() where
  # we pass a *.nvcpp file in order to force using the NVCXX compiler
  set(CMAKE_NVCXX_COMPILE_OBJECT "<CMAKE_NVCXX_COMPILER> -x c++ <DEFINES> <INCLUDES> <FLAGS> -o <OBJECT> -c <SOURCE>")
endif()

if(NOT CMAKE_NVCXX_LINK_EXECUTABLE)
  set(CMAKE_NVCXX_LINK_EXECUTABLE
      "<CMAKE_NVCXX_COMPILER> <FLAGS> <CMAKE_NVCXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
endif()

mark_as_advanced(CMAKE_VERBOSE_MAKEFILE)

set(CMAKE_NVCXX_INFORMATION_LOADED 1)
