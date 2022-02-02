# MindQuantum Build binary wheels

If you would like to build binary wheels of MindQuantum and then redistribute them to other people, you might want to consider delocating the wheels to make sure that no external dependencies linked to your particular system remain.

Currently, only Linux and MacOS are supported.

## Building the wheels

In order to generate binary wheels of MindQuantum, we recommend that you use the `build` package from Pypa:

    cd mindquantum
    python3 -m build .

This will, however, produce binary wheels that may depend on some external libraries found somewhere on your system. Distributing them might therefore run into some issues if the other users do not have the same libraies installed or in different non-standard locations.

##  Delocating the wheels

In order to make sure that all the required external library dependencies are contained within the binary wheel file, you can instruct the binary wheel building process to _delocate_ the produced wheels. This is achieved by specifying the `MQ_DELOCATE_WHEEL` environment variable, like so:

    cd mindquantum
    MQ_DELOCATE_WHEEL=1 python3 -m build .

On Linux, you may also specify which platform you are targeting by using the `MQ_DELOCATE_WHEEL_PLAT` environment variable. By default it assumes 'linux_x86_64' on 64 bits machines but you may specify any platforms supported by [auditwheel](https://github.com/pypa/auditwheel).
