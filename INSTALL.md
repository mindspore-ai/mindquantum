# Installation

This file is designed to get you started with building and installing MindQuantum.

## Quick-start

### Build a binary Python wheel

In order to generate binary wheels of MindQuantum, we recommend that you use the `build` package from Pypa:

```bash
cd mindquantum
python3 -m build .
```

This will, however, produce binary wheels that may depend on some external libraries found somewhere on your system. Distributing them might therefore run into some issues if the other users do not have the same libraries installed or in different non-standard locations.

#### Delocating the wheels

In order to make sure that all the required external library dependencies are contained within the binary wheel file, you can instruct the binary wheel building process to _delocate_ the produced wheels. In practice, this means to remove any dependency on external (shared) libraries on your system by integrating those directly into the binary wheel.

##### Using cibuildwheel

This is the preferred way of building binary wheels as it relies on Docker images (on Linux only) or the standard Python distribution on MacOS and Windows.

```bash
cd mindquantum
python3 -m cibuildwheel .
```

you might need to specify your platform if `cibuildwheel` is not able to automatically detect your platform or if you want to build Linux wheel on MacOS for example:

```bash
cd mindquantum
python3 -m cibuildwheel --platform linux .
```

See `python3 -m cibuildwheel --help` for more information about which platforms are available.

###### Linux

On Linux, you may run the script directly on any machine as it is using Docker images in order to build the delocated binary wheel. Same thing on MacOS or Windows if you would like to build Linux binary wheels.

###### MacOS

On MacOS, cibuildwheel will install the official Python distribution on your system before building the binary wheel. This makes running the script on your development machine not appropriate unless you understand what you are doing.

###### Windows

On Windows, cibuildwheel will install the official Python distribution using NuGet on your system before building the binary wheel. This makes running the script on your development machine not appropriate unless you understand what you are doing.

##### On your local machine

If you do not want to rely on the `cibuildwheel` machinery (e.g. on MacOS) you can also automatically call `auditwheel` or `delocate` after building the wheel by specifying the `MQ_DELOCATE_WHEEL` environment variable, like so:

```bash
cd mindquantum
MQ_DELOCATE_WHEEL=1 python3 -m build .
```

If you plan on distributing the wheel to other people that might not have the same system as yours, we highly recommend that you try to specify the `MQ_DELOCATE_WHEEL_PLAT` environment variable. By default, the setup scripts assumes 'linux_x86_64' on 64 bits machines but you may specify any platforms supported by [auditwheel](https://github.com/pypa/auditwheel). In order to distribute your wheels to a larger audience, we would recomment setting `MQ_DELOCATE_WHEEL_PLAT=manylinux2010_x86_64`, although this might result in an error when delocating the wheel if the version of your compiler is too recent.

### Build MindQuantum locally

You can setup MindQuantum for local development by using one of the local build scripts:

- `build_locally.bat` (MS-DOS BATCH script)
- `build_locally.ps1` (PowerShell script)
- `build_locally.sh` (Bash script)

Except a few minor differences [1]_, the functionalities of all three scripts are identical. All the scripts accept a flag to display the help message (``-h, --help``, ``-H, -Help`, ``/h, /Help`` for Bash, Powershell and MS-DOS BATCH). Please invoke the script of your choice in order to view the latest set of functionalities provided by it.

1. Setup a Python virtual environment
2. Update the virtual environment's packages and install some required dependencies
3. Add a PTH-file to the Python virtual environment to make sure that MindQuantum will be detected
4. Create a build directory and run CMake within it
5. Compile MindQuantum in-place

The next time you run the script, unless you specify one of the cleaning options or force a CMake configuration step,
the script will only re-compile MindQuantum.

For more information, have a look at the help message of the scripts which you can access using `./build_locally.sh -h` or `./build_locally.sh --help`. The output is shown below for reference.

## CMake configuration

### CMake options

Here is an exhaustive list of all CMake options available for customization

| Option name                     | Description                                                           | Default value       |
|---------------------------------|-----------------------------------------------------------------------|---------------------|
| BUILD_SHARED_LIBS               | Build shared libs                                                     | OFF                 |
| BUILD_TESTING                   | Enable building the test suite                                        | OFF                 |
| CLEAN_3RDPARTY_INSTALL_DIR      | Clean third-party installation directory                              | OFF                 |
| CUDA_ALLOW_UNSUPPORTED_COMPILER | Allow the use of an unsupported comipler version for CUDA             | OFF                 |
| CUDA_STATIC                     | Use static versions of the Nvidia CUDA libraries                      | OFF                 |
| DISABLE_FORTRAN_COMPILER        | Forcefully disable the Fortran compiler for some 3rd party libraries  | ON                  |
| ENABLE_CMAKE_DEBUG              | Enable verbose output to debug CMake issues                           | OFF                 |
| ENABLE_CUDA                     | Enable the use of CUDA code                                           | OFF                 |
| ENABLE_GITEE                    | Use Gitee instead of GitHub for (some) third-party dependencies       | OFF                 |
| ENABLE_MD                       | Use /MD, /MDd flags when compiling (MSVC only)                        | OFF                 |
| ENABLE_MT                       | Use /MT, /MTd flags when compiling (MSVC only)                        | OFF                 |
| ENABLE_PROFILING                | Enable compilation with profiling flags                               | OFF                 |
| ENABLE_RUNPATH                  | Prefer RUNPATH over RPATH when linking                                | ON                  |
| ENABLE_STACK_PROTECTION         | Enable stack protection during compilation                            | ON                  |
| IN_PLACE_BUILD                  | Build the C++ MindQuantum libraries in-place                          | OFF                 |
| IS_PYTHON_BUILD                 | Whether CMake is called from setup.py                                 | OFF                 |
| LINKER_DTAGS                    | Enable --enable-new-dtags (else use --disable-new-dtags) when linking | ON                  |
| LINKER_NOEXECSTACK              | Use `-z,noexecstack` during linking (if supported)                    | ON                  |
| LINKER_RELRO                    | Use `-z,relro` during linking (if supported)                          | ON                  |
| LINKER_RPATH                    | Use RUNPATH/RPATH related flags when compiling                        | ON                  |
| LINKER_STRIP_ALL                | Use `--strip-all` during linking (if supported)                       | ON                  |
| USE_OPENMP                      | Use the OpenMP library for parallelisation                            | ON                  |
| USE_PARALLEL_STL                | Use the parallel STL for parallelisation (using TBB or else)          | OFF                 |
| USE_VERBOSE_MAKEFILE            | Generate verbose Makefiles (if supported)                             | ON                  |

Here are some more precisions on some of the above options:

#### `CLEAN_3RDPARTY_INSTALL_DIR`

This will delete any pre-existing installations within the local installation directory (by default `/path/to/build/.mqlibs`) _except_ the ones that are currently needed based on the hashes of the third-party libraries.

#### `DISABLE_FORTRAN_COMPILER`

This currently only has an effect when installing Eigen3.

### CMake variables

In addition to the above CMake options, you may pass certain special CMake variables in order to customize your build. These are described below in more details.

#### `MQ_FORCE_LOCAL_PKGS`

This variable's value is case-insensitive. It may be either of:

- a single string (`all`)
- a comma-separated list of CMake package names for one or more of MindQuantum's third-party dependencies (e.g. `gmp,eigen3`)

Any or all packages listed will be compiled locally during the CMake configuration process.

#### `MQ_XXX_FORCE_LOCAL`

Setting this to a truthful value for one of MindQuantum's third-party dependencies will result in that/these packages to be compiled locally during the CMake configuration process. Note that the package name `XXX` must be all caps.
