# MindQuantum Build binary wheels

If you would like to build binary wheels of MindQuantum and then redistribute them to other people, you might want to consider delocating the wheels to make sure that no external dependencies linked to your particular system remain.

Currently, only Linux and MacOS are supported.

## Building the wheels

In order to generate binary wheels of MindQuantum, we recommend that you use the `build` package from Pypa:

```bash
cd mindquantum
python3 -m build .
```

This will, however, produce binary wheels that may depend on some external libraries found somewhere on your system. Distributing them might therefore run into some issues if the other users do not have the same libraries installed or in different non-standard locations.

## Delocating the wheels

In order to make sure that all the required external library dependencies are contained within the binary wheel file, you can instruct the binary wheel building process to _delocate_ the produced wheels. In practice, this means to remove any dependency on external (shared) libraries on your system by integrating those directly into the binary wheel.

### Using cibuildwheel

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

#### Linux

On Linux, you may run the script directly on any machine as it is using Docker images in order to build the delocated binary wheel. Same thing on MacOS or Windows if you would like to build Linux binary wheels.

#### MacOS

On MacOS, cibuildwheel will install the official Python distribution on your system before building the binary wheel. This makes running the script on your development machine not appropriate unless you understand what you are doing.

#### Windows

On Windows, cibuildwheel will install the official Python distribution using NuGet on your system before building the binary wheel. This makes running the script on your development machine not appropriate unless you understand what you are doing.

### On your local machine

If you do not want to rely on the `cibuildwheel` machinery (e.g. on MacOS) you can also automatically call `auditwheel` or `delocate` after building the wheel by specifying the `MQ_DELOCATE_WHEEL` environment variable, like so:

```bash
cd mindquantum
MQ_DELOCATE_WHEEL=1 python3 -m build .
```

If you plan on distributing the wheel to other people that might not have the same system as yours, we highly recommend that you try to specify the `MQ_DELOCATE_WHEEL_PLAT` environment variable. By default, the setup scripts assumes 'linux_x86_64' on 64 bits machines but you may specify any platforms supported by [auditwheel](https://github.com/pypa/auditwheel). In order to distribute your wheels to a larger audience, we would recomment setting `MQ_DELOCATE_WHEEL_PLAT=manylinux2010_x86_64`, although this might result in an error when delocating the wheel if the version of your compiler is too recent.
