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


.. _installation:

Quick start
===========

If you are only looking in using MindQuantum on your system, the easiest way of getting started is installing it
directly from Pypi and use one of the pre-compiled binaries: :ref:`install_from_pypi`.

If you are looking into doing some development with MindQuantum on your local machine, we highly recommend you using one
of the scripts provided to build MindQuantum locally: :ref:`build_locally_for_devs`. In that case, you might want to
install some of the required programs and libraries. See one of the sub-sections under :ref:`requirements` for your
particular system for more information in order to find out how to achieve that. Additionally, if you plan to link some
libraries you are developping to MindQuantum, have a look at the :ref:`install_locally`. This will guide you into adding
MindQuantum as a third-party library into your other projects.

If you plan on distributing the version of MindQuantum you have compiled on your system, we would suggest that you
have a look at :ref:`build_wheels`.

.. _install_from_pypi:

Install from Pypi
-----------------

You can install one of the pre-compiled binary Python packages directly from Pypi using Pip:

.. code-block:: bash

   python3 -m pip install --user mindquantum

.. _build_locally_for_devs:

Build locally (for developers)
------------------------------

In order to build MindQuantum locally for developping new features or implementing bug fixes for MindQuantum, there are
a few scripts that you can use to properly setup a virtual environment and all the required build tools (such as
CMake). Currently, there are three local build scripts:

- ``build_locally.bat`` (MS-DOS BATCH script)
- ``build_locally.ps1`` (PowerShell script)
- ``build_locally.sh`` (Bash script)

Except a few minor differences [1]_, the functionalities of all three scripts are identical. All the scripts accept a
flag to display the help message (``-h, --help``, ``-H, -Help`, ``/h, /Help`` for Bash, Powershell and MS-DOS
BATCH). Please invoke the script of your choice in order to view the latest set of functionalities provided by it.

The build scripts mentioned above will perform the following operations in order:

1. Setup a Python virtual environment
2. Update the virtual environment's packages and install some required dependencies (which may include CMake)
3. Add a PTH-file to the Python virtual environment to make sure that MindQuantum will be detected
4. Create a build directory and run CMake within it
5. Compile MindQuantum in-place

The next time you run the script, unless you specify one of the cleaning options or force a CMake configuration step,
the script will only re-compile MindQuantum.

For reference, here is the output of the help message from the Bash script (NB: might differ from the actual help
message):

.. code-block:: bash

  Build MindQunantum locally (in-source build)

  This is mainly relevant for developers that do not want to always have to reinstall the Python
  package

  This script will create a Python virtualenv in the MindQuantum root directory and then build
  all the C++ Python modules and place the generated libraries in their right locations within
  the MindQuantum folder hierarchy so Python knows how to find them.

  A pth-file will be created in the virtualenv site-packages directory so that the MindQuantum
  root folder will be added to the Python PATH without the need to modify PYTHONPATH.

  Usage:
    build_locally.sh [options] [-- cmake_options]

  Options:
    -h,--help            Show this help message and exit
    -n                   Dry run; only print commands but do not execute them

    -B,--build=[dir]     Specify build directory
                         Defaults to: /home/user/mindquantum/build
    --ccache             If ccache or sccache are found within the PATH, use them with CMake
    --clean-3rdparty     Clean 3rd party installation directory
    --clean-all          Clean everything before building.
                         Equivalent to --clean-venv --clean-builddir
    --clean-builddir     Delete build directory before building
    --clean-cache        Re-run CMake with a clean CMake cache
    --clean-venv         Delete Python virtualenv before building
    --config=[dir]       Path to INI configuration file with default values for the parameters
                         Defaults to: /home/user/mindquantum/build.conf
                         NB: command line arguments always take precedence over configuration
                         file values
    --debug              Build in debug mode
    --debug-cmake        Enable debugging mode for CMake configuration step
    --gpu                Enable GPU support
    -j,--jobs [N]        Number of parallel jobs for building
                         Defaults to: 16
    --local-pkgs         Compile third-party dependencies locally
    --ninja              Build using Ninja instead of make
    --quiet              Disable verbose build rules
    --show-libraries     Show all known third-party libraries
    -v, --verbose        Enable verbose output from the Bash scripts
    --venv=[dir]         Path to Python virtual environment
                         Defaults to: /home/user/mindquantum/venv
    --with-<library>     Build the third-party <library> from source
                         (ignored if --local-pkgs is passed, except for projectq)
    --without-<library>  Do not build the third-party library from source
                         (ignored if --local-pkgs is passed, except for projectq)

  Test related options:
    --test               Build C++ tests and install dependencies for Python testing as well
    --only-pytest        Only install pytest and its dependencies when creating/building the
                         virtualenv

  CUDA related options:
    --cuda-arch=[arch]   Comma-separated list of architectures to generate device code for.
                         Only useful if --gpu is passed. See CMAKE_CUDA_ARCHITECTURES for more
                         information.

  Python related options:
    --update-venv        Update the python virtual environment

  Developer options:
    --cmake-no-registry  Disable the use of CMake package registries during configuration

  Extra options:
    --clean              Run make clean before building
    -c,--configure       Force running the CMake configure step
    --configure-only     Stop after the CMake configure and generation steps
                         (ie. before building MindQuantum)
    --doc,--docs         Setup the Python virtualenv for building the documentation and ask
                         CMake to build the documentation
    --install            Build the ´install´ target
    --prefix             Specify installation prefix

  Any options after "--" will be passed onto CMake during the configuration step

  Example calls:
  build_locally.sh -B build
  build_locally.sh -B build --gpu
  build_locally.sh -B build --cxx --with-boost --without-gmp --venv=/tmp/venv
  build_locally.sh -B build -- -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc
  build_locally.sh -B build --cxx --gpu -- \
         -DCMAKE_NVCXX_COMPILER=/opt/nvidia/hpc_sdk/Linux_x86_64/22.3/compilers/bin/nvc++

.. [1] PowerShell and Bash scripts typically have identical functionality sets whereas the MS-DOS BATCH script might
       not. For example, the latter does not support the ``/WithOutXXX``-type arguments.

.. _install_locally:

Install locally (as a library)
------------------------------

If you plan on integrating MindQuantum into your own project as a third-party library, you may want to install it
locally on your computer. For that you may use the scripts mentioned in Section :ref:`build_locally_for_devs`.

There are essentially three ways you can include MindQuantum as a third-party library:

1. As a sub-directory if your project also uses CMake (``add_subdirectory("path/to/mindquantum")``)
2. Installing MindQuantum as a library somewhere on your system
3. Using the build directory as a pseudo-installation location (provided your project also uses CMake)

As option 1. is pretty straightforward, we will not provide more explanation here. However, for the other two options,
some more detailed help can be found below.

Installation on your system
+++++++++++++++++++++++++++

If you are using the local build scripts, simply add the ``--install`` (or ``-Install`` or ``/Install``) and if
necessary the ``--prefix`` (or ``-Prefix`` or ``/Prefix``) arguments to your command line to install MindQuantum in your
preferred location.

Given an installation ``<prefix>``, building the ``install`` target will result in the relevant files being installed
into:

``<prefix>/include/mindquantum``
    All MindQuantum header files

``<prefix>/lib/mindquantum``
    All MindQuantum libraries (excuding 3rd-party libraries)

``<prefix>/lib/mindquantum/third_party``
    All 3rd-party libraries (including their header files). This is actually the content of the ``build/.mqlibs`` folder
    within the build directory.

``<prefix>/share/cmake/mindquantum/``
    All CMake installation configuration files. This includes the ``mindquantumConfig.cmake`` file and other helper
    files.

Then, in order to use MindQuantum in some other CMake project, you simply need to add the following statement:
``find_package(mindquantum CONFIG)``:

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.20)
    project(XXXX CXX)

    # ...

    find_package(mindquantum CONFIG)

    # ...


You may need to also define either of ``mindquantum_ROOT`` or ``mindquantum_DIR`` CMake variables in order to help CMake
locate the MindQuantum installation. For the former, simply defining it to ``<prefix>`` should suffice:

.. code-block:: bash

    cmake ... -Dmindquantum_ROOT=/path/to/mindquantum/install

Instead of defining ``mindquantum_ROOT`` you may alternatively define ``mindquantum_DIR``. In this case, the path must
be to the directory that contains the ``mindquantumConfig.cmake`` file.

.. code-block:: bash

    cmake ... -Dmindquantum_DIR=/path/to/mindquantum/install/share/mindquantum/cmake

.. note::

    You may defined either of ``mindquantum_DEBUG`` or ``MINDQUANTUM_DEBUG`` to a truthful value to have CMake be more
    verbose when reading the MindQuantum configuration files.

If you have MindQuantum installed as a Python package, you can also use the module itself to locate where the CMake
installation config file is located. You may use any of the following commands:

.. code-block:: bash

    > python3 -m mindquantum --cmakedir
    /usr/local/lib/python3.10/site-packages/mindquantum/share/mindquantum/cmake

    > mindquantum-config --cmakedir
    /usr/local/lib/python3.10/site-packages/mindquantum/share/mindquantum/cmake

.. note::

    Both of the above commands provide the exact same information. The advantage of the latter over the former is that
    it does not attempt to load the mindquantum package which may be slower to execute in practice.

In-build "pseudo"-install
+++++++++++++++++++++++++

If you do not wish to install MindQuantum on your system, you may use the build directory as a *pseudo-installation*
location. Simply follow the above instructions and simply set either of ``mindquantum_ROOT`` or ``mindquantum_DIR``
CMake variables to point to your build directory:

.. code-block:: bash

    cmake ... -Dmindquantum_DIR=/path/to/mindquantum/build


.. _build_wheels:

Build binary Python wheels (for distribution)
---------------------------------------------

If you plan on compiling MindQuantum on your local machine (or some CI) and would like to distribute the code in binary
form to other users, we woul dsuggest you take a look at the ``build.sh`` script.

The build script mentioned above will perform the following operations in order:

1. Setup a Python virtual environment
2. Update the virtual environment's packages and install some required dependencies (which may include CMake)
3. Call ``python3 -m build``


It has similar options as the scripts described in :ref:`build_locally_for_devs`:

.. code-block::

  Build binary Python wheel for MindQunantum

  This is mainly relevant for developers that want to deploy MindQuantum on machines other
  than their own.

  This script will create a Python virtualenv in the MindQuantum root directory and then build a
  binary Python wheel of MindQuantum.

  Usage:
    build.sh [options] [-- cmake_options]

  Options:
    -h,--help            Show this help message and exit
    -n                   Dry run; only print commands but do not execute them

    -B,--build=[dir]     Specify build directory
                         Defaults to: /home/user/mindquantum/build
    --ccache             If ccache or sccache are found within the PATH, use them with CMake
    --clean-3rdparty     Clean 3rd party installation directory
    --clean-all          Clean everything before building.
                         Equivalent to --clean-venv --clean-builddir
    --clean-builddir     Delete build directory before building
    --clean-cache        Re-run CMake with a clean CMake cache
    --clean-venv         Delete Python virtualenv before building
    --config=[dir]       Path to INI configuration file with default values for the parameters
                         Defaults to: /home/user/mindquantum/build.conf
                         NB: command line arguments always take precedence over configuration
                         file values
    --debug              Build in debug mode
    --debug-cmake        Enable debugging mode for CMake configuration step
    --gpu                Enable GPU support
    -j,--jobs [N]        Number of parallel jobs for building
                         Defaults to: 16
    --local-pkgs         Compile third-party dependencies locally
    --ninja              Build using Ninja instead of make
    --quiet              Disable verbose build rules
    --show-libraries     Show all known third-party libraries
    -v, --verbose        Enable verbose output from the Bash scripts
    --venv=[dir]         Path to Python virtual environment
                         Defaults to: /home/user/mindquantum/venv
    --with-<library>     Build the third-party <library> from source
                         (ignored if --local-pkgs is passed, except for projectq)
    --without-<library>  Do not build the third-party library from source
                         (ignored if --local-pkgs is passed, except for projectq)

  Test related options:
    --test               Build C++ tests and install dependencies for Python testing as well
    --only-pytest        Only install pytest and its dependencies when creating/building the
                         virtualenv

  CUDA related options:
    --cuda-arch=[arch]   Comma-separated list of architectures to generate device code for.
                         Only useful if --gpu is passed. See CMAKE_CUDA_ARCHITECTURES for more
                         information.

  Python related options:
    --update-venv        Update the python virtual environment

  Developer options:
    --cmake-no-registry  Disable the use of CMake package registries during configuration

  Extra options:
    --delocate           Delocate the binary wheels after build is finished
                         (enabled by default; pass --no-delocate to disable)
    --no-delocate        Disable delocating the binary wheels after build is finished
    --no-build-isolation Pass --no-isolation to python3 -m build
    -o,--output=[dir]    Output directory for built wheels
    -p,--plat-name=[dir] Platform name to use for wheel delocation
                         (only effective if --delocate is used)

  Example calls:
  build.sh
  build.sh --gpu
  build.sh --cxx --with-boost --without-gmp --venv=/tmp/venv

.. _requirements:

Requirements
============

.. toctree::
   :maxdepth: 2


In order to get started with MindQuantum, you will need to have a C++ compiler installed on your system as well as a few
libraries and programs:

    - Python >= 3.5
    - CMake >= 3.20

Below you will find detailed installation instructions for various operating systems.

Linux
-----

Ubuntu/Debian
+++++++++++++

After having installed the build tools (for g++):

.. code-block:: bash

   sudo apt-get install build-essential

You only need to install Python (and the package manager). For version 3.x, run

.. code-block:: bash

   sudo apt-get install python3-dev python3-pip python3-venv


If the CMake version provided by Ubuntu is not recent enough (typically the case for Ubuntu <= 21.10), you may want to
install CMake using Pip:

.. code-block:: bash

   python3 -m pip install --user cmake

Otherwise, install CMake using APT as normal:

.. code-block:: bash

   sudo apt-get install cmake

.. note::

   On Ubuntu, you may use the https://apt.kitware.com/ repository. Follow the instruction there in order to install the
   latest CMake using APT.

ArchLinux/Manjaro
+++++++++++++++++

Make sure that you have a C/C++ compiler installed:

.. code-block:: bash

   sudo pacman -Syu gcc

You only need to install Python (and the package manager). For version 3, run

.. code-block:: bash

   sudo pacman -Syu python python-pip

Then install CMake using the following command:

.. code-block:: bash

   sudo pacman -Syu cmake


CentOS 7
++++++++

Run the following commands:

.. code-block:: bash

   sudo yum install -y epel-release
   sudo yum install -y centos-release-scl
   sudo yum install -y devtoolset-8
   sudo yum check-update -y

   scl enable devtoolset-8 bash
   sudo yum install -y gcc-c++ make git
   sudo yum install -y python3 python3-devel python3-pip

   sudo python3 -m pip install cmake

CentOS 8
++++++++

Run the following commands:

.. code-block:: bash

   # The following two lines might not be required in all situations.
   sudo sed -i 's/mirrorlist/#mirrorlist/g' /etc/yum.repos.d/CentOS-*
   sudo sed -i 's|#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|g' /etc/yum.repos.d/CentOS-*

   sudo dnf config-manager --set-enabled PowerTools
   sudo yum install -y epel-release
   sudo yum check-update -y

   sudo yum install -y gcc-c++ make git
   sudo yum install -y python3 python3-devel python3-pip

   sudo python3 -m pip install cmake

Mac OS
------

We require that a C++ compiler is installed on your system. There are two options you can choose from:

   1. Using Homebrew
   2. Using MacPorts


Before moving on, install the XCode command line tools by opening a terminal window and running the following command:

.. code-block:: bash

   xcode-select --install

Homebrew
++++++++

Install Homebrew with the following command:

.. code-block:: bash

   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

Then proceed to install Python as well as a C++ compiler (note: gcc installed via Homebrew may lead to some issues
therefore we choose clang):

.. code-block:: bash

   brew install python llvm

Then install the rest of the required libraries/programs using the following command:

.. code-block:: bash

   sudo port install cmake


MacPorts
++++++++

Visit `macports.org <https://www.macports.org/install.php>`_ and install the latest version that corresponds to your
operating system's version. Afterwards, open a new terminal window.

Then, use macports to install Python 3.8 with pip by entering the following command

.. code-block:: bash

   sudo port install python38 py38-pip

A warning might show up about using Python from the terminal.In this case, you should also install

.. code-block:: bash

   sudo port install py38-gnureadline

Then install the rest of the required libraries/programs and a C++ compiler using the following command:

.. code-block:: bash

   sudo port install cmake clang-14


Windows
-------

On Windows, you may compile MindQuantum using any of the following methods:

- :ref:`msvc` 2019 or more recent
- :ref:`msys2` using either MSYS2-MSYS or MSYS2-MINGW64
- :ref:`cygwin`
- :ref:`mingw64`

While we cannot provide an exhaustive guide on how to compile using each of the aforementioned methods, you can use the
following as a starting point. Also note that most if not all of the above are testing using GitHub actions. When in
doubt, you may have a look at the workflow configuration file to see exactly how MindQunantum is compiled there.


.. _msvc:

Visual Studio
+++++++++++++

You may either install Visual Studio 2019 or more recent using the installer provided by Microsoft or use the
`Chocolatey package manager <https://chocolatey.org/>`_. Note that in some cases, the automatic build of the Boost
libraries during the CMake call might fail. In that case, we would suggest that you compile and install those libraries
separately and then attempt building MindQuantum again.

In the following, all the commands are to be run from within a PowerShell window. In some cases, you might need to run
PowerShell as administrator.

Pure Windows install
````````````````````

Install Python using the installer provided at https://www.python.org/downloads/.

Chocolatey
``````````

First install Chocolatey using the installer following the instructions on their `website
<https://chocolatey.org/docs/installation>`_. Once that is done, you can start by installing some of the required
packages. Reboot as needed during the process.

.. code-block:: powershell

   choco install -y visualstudio2019-workload-vctools --includeOptional
   choco install -y windows-sdk-10-version-2004-all
   choco install -y cmake git

Installing Python is as simply as running the following commands:

.. code-block:: powershell

   choco install -y python3 --version 3.9.11
   cmd /c mklink "C:\Python38\python3.exe" "C:\Python38\python.exe"

.. _msys2:

MSYS2
+++++

Install MSYS2 using the installer provided at https://www.msys2.org/.


MSYS2-MSYS
``````````

From within an MSYS2-MSYS shell, run the following command in order to install the required programs and libraries:

.. code-block:: bash

   pacman -Syu
   pacman -S git base-devel gcc cmake python-devel python-pip gmp-devel

MSYS2-MINGW64
`````````````

From within an MSYS2-MINGW64 shell, run the following command in order to install the required programs and libraries:

.. code-block:: bash

   pacman -Syu
   pacman -S git patch make mingw-w64-x86_64-toolchain mingw-w64-x86_64-cmake \
                mingw-w64-x86_64-python mingw-w64-x86_64-python-pip

.. note::

   When using MSYS2-MINGW64, you will need to use the "MSYS Makefiles" generator for CMake. Simply provide ``-G "MSYS
   Makefiles"`` on the command line as argument to CMake.

.. _cygwin:

Cygwin
++++++

Install Cygwin using the installer provided at https://www.cygwin.com/install.html.

Then install the following packages:

- autoconf
- automake
- binutils
- m4
- make
- cmake
- patch
- gzip
- bzip2
- tar
- xz
- flex
- file
- findutils
- groff
- gawk
- sed
- libtool
- gettext
- wget
- curl
- grep
- dos2unix
- git
- gcc-core
- gcc-g++
- libgmp-devel
- python3
- python3-devel
- python3-pip
- python3-virtualenv


.. _mingw64:

MinGW64
+++++++

Install MinGW64 by following the instructions at https://www.mingw-w64.org/downloads/.

You then will want to install Python; e.g. using the installer provided at https://www.python.org/downloads/ and then
install CMake using Pip:

.. code-block:: bash

   python -m pip install --user cmake
