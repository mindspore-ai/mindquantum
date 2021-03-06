include(Compiler/PGI)
__compiler_pgi(NVCXX)
string(APPEND CMAKE_NVCXX_FLAGS_MINSIZEREL_INIT " -DNDEBUG")
string(APPEND CMAKE_NVCXX_FLAGS_RELEASE_INIT " -DNDEBUG")

if(CMAKE_NVCXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12.10)
  set(CMAKE_NVCXX98_STANDARD_COMPILE_OPTION "")
  set(CMAKE_NVCXX98_EXTENSION_COMPILE_OPTION --gnu_extensions)
  set(CMAKE_NVCXX98_STANDARD__HAS_FULL_SUPPORT ON)
  if(CMAKE_NVCXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13.10)
    set(CMAKE_NVCXX98_STANDARD_COMPILE_OPTION --c++03)
    set(CMAKE_NVCXX98_EXTENSION_COMPILE_OPTION --c++03 --gnu_extensions)
    set(CMAKE_NVCXX11_STANDARD_COMPILE_OPTION --c++11)
    set(CMAKE_NVCXX11_EXTENSION_COMPILE_OPTION --c++11 --gnu_extensions)
    set(CMAKE_NVCXX11_STANDARD__HAS_FULL_SUPPORT ON)
    if(CMAKE_NVCXX_COMPILER_VERSION VERSION_GREATER_EQUAL 15.7)
      set(CMAKE_NVCXX14_STANDARD_COMPILE_OPTION --c++14)
      set(CMAKE_NVCXX14_EXTENSION_COMPILE_OPTION --c++14 --gnu_extensions)
      set(CMAKE_NVCXX14_STANDARD__HAS_FULL_SUPPORT ON)
      if(CMAKE_NVCXX_COMPILER_VERSION VERSION_GREATER_EQUAL 17.1)
        set(CMAKE_NVCXX17_STANDARD_COMPILE_OPTION --c++17)
        set(CMAKE_NVCXX17_EXTENSION_COMPILE_OPTION --c++17 --gnu_extensions)
      endif()
    endif()
  endif()
endif()

if(CMAKE_VERSION VERSION_LESS 3.22)
  set(CMAKE_NVCXX_STANDARD_COMPUTED_DEFAULT 14)
endif()
__compiler_check_default_language_standard(NVCXX 12.10 98)

include(Compiler/NVHPC)

# Needed so that we support `LANGUAGE` property correctly
set(CMAKE_NVCXX_COMPILE_OPTIONS_EXPLICIT_LANGUAGE -x c++)

if(CMAKE_NVCXX_COMPILER_VERSION VERSION_GREATER_EQUAL 20.11)
  set(CMAKE_NVCXX20_STANDARD_COMPILE_OPTION -std=c++20)
  set(CMAKE_NVCXX20_EXTENSION_COMPILE_OPTION -std=gnu++20)
endif()

if(CMAKE_NVCXX_COMPILER_VERSION VERSION_GREATER_EQUAL 21.07)
  set(CMAKE_DEPFILE_FLAGS_NVCXX "-MD -MT <DEP_TARGET> -MF <DEP_FILE>")
  set(CMAKE_NVCXX_DEPFILE_FORMAT gcc)
  set(CMAKE_NVCXX_DEPENDS_USE_COMPILER TRUE)
else()
  # Before NVHPC 21.07 the `-MD` flag implicitly implies `-E` and therefore compilation and dependency generation can't
  # occur in the same invocation
  set(CMAKE_NVCXX_DEPENDS_EXTRA_COMMANDS
      "<CMAKE_NVCXX_COMPILER> <DEFINES> <INCLUDES> -x c++ <FLAGS> -M <SOURCE> -MT <OBJECT> -MD<DEP_FILE>")
endif()
__compiler_nvhpc(NVCXX)
