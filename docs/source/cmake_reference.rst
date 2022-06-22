.. Copyright 2022 <Huawei Technologies Co., Ltd>

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

.. toctree::
   :maxdepth: 2


.. _cmake-reference:

CMake reference
===============

CMake options
-------------

Here is an exhaustive list of all CMake options available for customization.

Descriptions
++++++++++++

+-----------------------------------+-----------------------------------------------------------------------+
| Option name                       | Description                                                           |
+-----------------------------------+-----------------------------------------------------------------------+
| ``BUILD_SHARED_LIBS``             | Build shared libs                                                     |
+-----------------------------------+-----------------------------------------------------------------------+
| ``BUILD_TESTING``                 | Enable building the test suite                                        |
+-----------------------------------+-----------------------------------------------------------------------+
| ``CLEAN_3RDPARTY_INSTALL_DIR``    | Clean third-party installation directory                              |
+-----------------------------------+-----------------------------------------------------------------------+
|``CUDA_ALLOW_UNSUPPORTED_COMPILER``| Allow the use of an unsupported comipler version for CUDA             |
+-----------------------------------+-----------------------------------------------------------------------+
| ``CUDA_STATIC``                   | Use static versions of the Nvidia CUDA libraries                      |
+-----------------------------------+-----------------------------------------------------------------------+
| ``DISABLE_FORTRAN_COMPILER``      | Forcefully disable the Fortran compiler for some 3rd party libraries  |
+-----------------------------------+-----------------------------------------------------------------------+
| ``ENABLE_CMAKE_DEBUG``            | Enable verbose output to debug CMake issues                           |
+-----------------------------------+-----------------------------------------------------------------------+
| ``ENABLE_CUDA``                   | Enable the use of CUDA code                                           |
+-----------------------------------+-----------------------------------------------------------------------+
| ``ENABLE_CXX_EXPERIMENTAL``       | Enable the building of the (new) experimental C++ backend             |
+-----------------------------------+-----------------------------------------------------------------------+
| ``ENABLE_GITEE``                  | Use Gitee instead of GitHub for (some) third-party dependencies       |
+-----------------------------------+-----------------------------------------------------------------------+
| ``ENABLE_MD``                     | Use /MD, /MDd flags when compiling (MSVC only)                        |
+-----------------------------------+-----------------------------------------------------------------------+
| ``ENABLE_MT``                     | Use /MT, /MTd flags when compiling (MSVC only)                        |
+-----------------------------------+-----------------------------------------------------------------------+
| ``ENABLE_PROFILING``              | Enable compilation with profiling flags                               |
+-----------------------------------+-----------------------------------------------------------------------+
| ``ENABLE_PROJECTQ``               | Enable ProjectQ support                                               |
+-----------------------------------+-----------------------------------------------------------------------+
| ``ENABLE_RUNPATH``                | Prefer RUNPATH over RPATH when linking                                |
+-----------------------------------+-----------------------------------------------------------------------+
| ``ENABLE_STACK_PROTECTION``       | Enable stack protection during compilation                            |
+-----------------------------------+-----------------------------------------------------------------------+
| ``IN_PLACE_BUILD``                | Build the C++ MindQuantum libraries in-place                          |
+-----------------------------------+-----------------------------------------------------------------------+
| ``IS_PYTHON_BUILD``               | Whether CMake is called from setup.py                                 |
+-----------------------------------+-----------------------------------------------------------------------+
| ``LINKER_DTAGS``                  |       Enable --enable-new-dtags (else use --disable-new-dtags)        |
+-----------------------------------+-----------------------------------------------------------------------+
| ``LINKER_NOEXECSTACK``            | Use `-z,noexecstack` during linking (if supported)                    |
+-----------------------------------+-----------------------------------------------------------------------+
| ``LINKER_RELRO``                  | Use `-z,relro` during linking (if supported)                          |
+-----------------------------------+-----------------------------------------------------------------------+
| ``LINKER_RPATH``                  | Use RUNPATH/RPATH related flags when compiling                        |
+-----------------------------------+-----------------------------------------------------------------------+
| ``LINKER_STRIP_ALL``              | Use `--strip-all` during linking (if supported)                       |
+-----------------------------------+-----------------------------------------------------------------------+
| ``USE_OPENMP``                    | Use the OpenMP library for parallelisation                            |
+-----------------------------------+-----------------------------------------------------------------------+
| ``USE_PARALLEL_STL``              | Use the parallel STL for parallelisation (using TBB or else)          |
+-----------------------------------+-----------------------------------------------------------------------+
| ``USE_VERBOSE_MAKEFILE``          | Generate verbose Makefiles (if supported)                             |
+-----------------------------------+-----------------------------------------------------------------------+

Default values
++++++++++++++

+-----------------------------------+------------------------------+
| Option name                       | Default value                |
+-----------------------------------+------------------------------+
| ``BUILD_SHARED_LIBS``             | OFF                          |
+-----------------------------------+------------------------------+
| ``BUILD_TESTING``                 | OFF                          |
+-----------------------------------+------------------------------+
| ``CLEAN_3RDPARTY_INSTALL_DIR``    | OFF                          |
+-----------------------------------+------------------------------+
|``CUDA_ALLOW_UNSUPPORTED_COMPILER``| OFF                          |
+-----------------------------------+------------------------------+
| ``CUDA_STATIC``                   | OFF                          |
+-----------------------------------+------------------------------+
| ``DISABLE_FORTRAN_COMPILER``      | ON                           |
+-----------------------------------+------------------------------+
| ``ENABLE_CMAKE_DEBUG``            | OFF                          |
+-----------------------------------+------------------------------+
| ``ENABLE_CUDA``                   | OFF                          |
+-----------------------------------+------------------------------+
| ``ENABLE_CXX_EXPERIMENTAL``       | OFF                          |
+-----------------------------------+------------------------------+
| ``ENABLE_GITEE``                  | OFF                          |
+-----------------------------------+------------------------------+
| ``ENABLE_MD``                     | OFF                          |
+-----------------------------------+------------------------------+
| ``ENABLE_MT``                     | OFF                          |
+-----------------------------------+------------------------------+
| ``ENABLE_PROFILING``              | OFF                          |
+-----------------------------------+------------------------------+
| ``ENABLE_PROJECTQ``               | ON                           |
+-----------------------------------+------------------------------+
| ``ENABLE_RUNPATH``                | ON                           |
+-----------------------------------+------------------------------+
| ``ENABLE_STACK_PROTECTION``       | ON                           |
+-----------------------------------+------------------------------+
| ``IN_PLACE_BUILD``                | OFF                          |
+-----------------------------------+------------------------------+
| ``IS_PYTHON_BUILD``               | OFF                          |
+-----------------------------------+------------------------------+
| ``LINKER_DTAGS``                  | ON                           |
+-----------------------------------+------------------------------+
| ``LINKER_NOEXECSTACK``            | ON                           |
+-----------------------------------+------------------------------+
| ``LINKER_RELRO``                  | ON                           |
+-----------------------------------+------------------------------+
| ``LINKER_RPATH``                  | ON                           |
+-----------------------------------+------------------------------+
| ``LINKER_STRIP_ALL``              | ON                           |
+-----------------------------------+------------------------------+
| ``USE_OPENMP``                    | ON                           |
+-----------------------------------+------------------------------+
| ``USE_PARALLEL_STL``              | OFF                          |
+-----------------------------------+------------------------------+
| ``USE_VERBOSE_MAKEFILE``          | ON                           |
+-----------------------------------+------------------------------+

Detailed descriptions
+++++++++++++++++++++

``CLEAN_3RDPARTY_INSTALL_DIR``
    This will delete any pre-existing installations within the local installation directory (by default
    ``/path/to/build/.mqlibs``) _except_ the ones that are currently needed based on the hashes of the third-party
    libraries.

``DISABLE_FORTRAN_COMPILER``
    This currently only has an effect when installing Eigen3.

CMake variables
---------------

In addition to the above CMake options, you may pass certain special CMake variables in order to customize your
build. These are described below in more details.


``MQ_FORCE_LOCAL_PKGS``
    This variable value is case-insensitive. It may be either of:
      - a single string (``all``)
      - a comma-separated list of CMake package names for one or more of MindQuantum's third-party dependencies
        (e.g. ``gmp,eigen3``)

    Any or all packages listed will be compiled locally during the CMake configuration process.

``MQ_XXX_FORCE_LOCAL``
    Setting this to a truthful value for one of MindQuantum's third-party dependencies will result in that/these
    packages to be compiled locally during the CMake configuration process. Note that the package name ``XXX`` must be
    all caps.
