# -*- coding: utf-8 -*-
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Setup.py file."""

import contextlib
import copy
import distutils.log
import errno
import itertools
import multiprocessing
import os
import platform
import shutil
import stat
import subprocess
import sys
from distutils.cmd import Command
from distutils.command.clean import clean
from distutils.file_util import copy_file

import setuptools
from setuptools.command.build_ext import build_ext

# ==============================================================================
# Helper variables

on_rtd = os.environ.get('READTHEDOCS') == 'True'
cur_dir = os.path.dirname(os.path.realpath(__file__))
ext_errors = (subprocess.CalledProcessError, FileNotFoundError)
cmake_extra_options = []

# ==============================================================================
# Helper functions and classes


@contextlib.contextmanager
def fdopen(fname, mode, perms=0o644):  # pragma: no cover
    """
    Context manager for opening files with correct permissions.

    Args:
        fname (str): Path to file to open for reading/writing
        mode (str): Mode in which the file is opened (see help for builtin `open()`)
        perms (int): Permission mask (see help for `os.open()`)
    """
    if 'r' in mode:
        flags = os.O_RDONLY
    elif 'w' in mode:
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    elif 'a' in mode:
        flags = os.O_WRONLY | os.O_CREAT
    else:
        raise RuntimeError(f'Unsupported mode: {mode}')

    file_object = open(os.open(fname, flags, perms), mode=mode, encoding='utf-8')

    try:
        yield file_object
    finally:
        file_object.close()


def remove_tree(directory):
    """Remove a directory and its subdirectories."""

    def remove_read_only(func, path, exc_info):
        excvalue = exc_info[1]
        if func in (os.rmdir, os.remove) and excvalue.errno == errno.EACCES:
            os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            func(path)
        else:
            raise exc_info[0].with_traceback(exc_info[1], exc_info[2])

    if os.path.exists(directory):
        distutils.log.info(f'Removing {directory} (and everything under it)')
        shutil.rmtree(directory, ignore_errors=False, onerror=remove_read_only)


def important_msgs(*msgs):
    """Print an important message."""
    print('*' * 75)
    for msg in msgs:
        print(msg)
    print('*' * 75)


def get_extra_cmake_options():
    """
    Parse CMake options from python3 setup.py command line.

    Read --unset, --set, -A and -G options from the command line and add them as cmake switches.
    """
    _cmake_extra_options = []

    opt_key = None

    has_generator = False

    argv = copy.deepcopy(sys.argv)
    # parse command line options and consume those we care about
    for arg in argv:
        if opt_key == 'G':
            has_generator = True
            _cmake_extra_options += ['-G', arg.strip()]
        elif opt_key == 'A':
            _cmake_extra_options += ['-A', arg.strip()]
        elif opt_key == 'unset':
            _cmake_extra_options.append(f'-D{arg.strip()}:BOOL=OFF')
        elif opt_key == 'set':
            _cmake_extra_options.append(f'-D{arg.strip()}:BOOL=ON')

        if opt_key:
            sys.argv.remove(arg)
            opt_key = None
            continue

        if arg in ['--unset', '--set', '--compiler-flags']:
            opt_key = arg[2:].lower()
            sys.argv.remove(arg)
            continue
        if arg in ['-A']:
            opt_key = arg[1:]
            sys.argv.remove(arg)
            continue
        if arg in ['-G']:
            opt_key = arg[1:]
            sys.argv.remove(arg)
            continue

    # If no explicit CMake Generator specification, prefer MinGW Makefiles on Windows
    if (not has_generator) and (platform.system() == "Windows"):
        _cmake_extra_options += ['-G', "MinGW Makefiles"]

    return _cmake_extra_options


# ==============================================================================


def get_executable(exec_name):
    """Try to locate an executable in a Python virtual environment."""
    try:
        root_path = os.environ['VIRTUAL_ENV']
        python = os.path.basename(sys.executable)
    except KeyError:
        root_path, python = os.path.split(sys.executable)

    exec_name = os.path.basename(exec_name)

    distutils.log.info(f'trying to locate {exec_name} in {root_path}')

    search_paths = [root_path, os.path.join(root_path, 'bin'), os.path.join(root_path, 'Scripts')]

    # First try executing the program directly
    for base_path in search_paths:
        try:
            cmd = os.path.join(base_path, exec_name)
            with fdopen(os.devnull, 'w') as devnull:
                subprocess.check_call([cmd, '--version'], stdout=devnull, stderr=devnull)
        except (OSError, subprocess.CalledProcessError):
            distutils.log.info(f'  failed in {base_path}')
        else:
            distutils.log.info(f'  command found: {cmd}')
            return cmd

    # That did not work: try calling it through Python
    for base_path in search_paths:
        try:
            cmd = [python, os.path.join(base_path, exec_name)]
            with fdopen(os.devnull, 'w') as devnull:
                subprocess.check_call(cmd + ['--version'], stdout=devnull, stderr=devnull)
        except (OSError, subprocess.CalledProcessError):
            distutils.log.info(f'  failed in {base_path}')
        else:
            distutils.log.info(f'  command found: {cmd}')
            return cmd

    distutils.log.info(f'  command *not* found!')
    return None


def get_python_executable():
    """Retrieve the path to the Python executable."""
    return get_executable(sys.executable)


def get_cmake_command():
    """Retrieve the path to the CMake executable."""
    try:
        with fdopen(os.devnull, 'w') as devnull:
            subprocess.check_call(['cmake', '--version'], stdout=devnull, stderr=devnull)
        return 'cmake'
    except (OSError, subprocess.CalledProcessError):
        pass

    # CMake not in PATH, should have installed the Python CMake module
    # -> try to find out where it is
    return get_executable('cmake')


# ==============================================================================


class BuildFailed(Exception):
    """Extension raised if the build fails for any reason."""

    def __init__(self):
        """Initialize a BuildFailed exception."""
        super().__init__()
        self.cause = sys.exc_info()[1]  # work around py 2/3 different syntax


# ==============================================================================


class CMakeExtension(setuptools.Extension):  # pylint: disable=too-few-public-methods
    """Class defining a C/C++ Python extension to be compiled using CMake."""

    def __init__(self, pymod, target=None, optional=False):
        """
        Initialize a CMakeExtension object.

        Args:
            src_dir (string): Path to source directory
            target (string): Name of target
            pymod (string): Name of compiled Python module
            optional (bool): (optional) If true, not building this extension is not considered an error
        """
        # NB: the main source directory is the one containing the setup.py file
        self.src_dir = os.path.realpath('')
        self.pymod = pymod
        self.target = target if target is not None else pymod.split('.')[-1]

        self.lib_filepath = os.path.join(*pymod.split('.'))
        super().__init__(pymod, sources=[], optional=optional)


# ------------------------------------------------------------------------------


class CMakeBuildExt(build_ext):
    """Custom build_ext command class."""

    user_options = build_ext.user_options + [
        ('no-arch-native', None, 'Do not use the -march=native flag when compiling'),
        ('clean-build', None, 'Build in a clean build environment'),
    ]

    boolean_options = build_ext.boolean_options + ['no-arch-native', 'clean-build']

    def initialize_options(self):
        """Initialize all options of this custom command."""
        build_ext.initialize_options(self)
        self.no_arch_native = None
        self.clean_build = None

    def build_extensions(self):
        """Build a C/C++ extension using CMake."""
        # pylint: disable=attribute-defined-outside-init
        if on_rtd:
            important_msgs('skipping CMake build on ReadTheDocs')
            return
        cmake_cmd = get_cmake_command()
        if cmake_cmd is None:
            raise RuntimeError('Unable to locate the CMake command!')
        self.cmake_cmd = [cmake_cmd]
        distutils.log.info('using cmake command: ' + ' '.join(self.cmake_cmd))

        self.configure_extensions()
        build_ext.build_extensions(self)

    def configure_extensions(self):
        """Run a CMake configuration and generation step for one extension."""
        # pylint: disable=attribute-defined-outside-init

        def _src_dir_pred(ext):
            return ext.src_dir

        cmake_args = [
            '-DPython_EXECUTABLE:FILEPATH=' + get_python_executable(),
            '-DBUILD_TESTING:BOOL=OFF',
            '-DIN_PLACE_BUILD:BOOL=OFF',
            '-DIS_PYTHON_BUILD:BOOL=ON',
            '-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON',
            '-DVERSION_INFO="{self.distribution.get_version()}"',
        ]  # yapf: disable

        if self.no_arch_native:
            cmake_args += ['-DUSE_NATIVE_INTRINSICS=OFF']

        cfg = 'Debug' if self.debug else 'Release'
        self.build_args = ['--config', cfg]

        if platform.system() == "Windows":
            # self.build_args += ['--', '/m']
            pass
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            if platform.system() == "Darwin" and 'TRAVIS' in os.environ:
                self.build_args += ['--']
            else:
                self.build_args += [
                    f'-j {self.parallel if self.parallel else multiprocessing.cpu_count()}',
                    '--',
                ]

        cmake_args.extend(cmake_extra_options)

        env = os.environ.copy()

        # This can in principle handle the compilation of extensions outside the main CMake directory (ie. outside the
        # one containing this setup.py file)
        for src_dir, extensions in itertools.groupby(sorted(self.extensions, key=_src_dir_pred), key=_src_dir_pred):
            self.cmake_configure_build(src_dir, extensions, cmake_args, env)

    def cmake_configure_build(self, src_dir, extensions, cmake_args, env):
        """Run a CMake build command for a list of extensions."""
        args = cmake_args.copy()
        for ext in extensions:
            dest_path = os.path.realpath(os.path.dirname(self.get_ext_fullpath(ext.lib_filepath)))
            args.append(f'-D{ext.target.upper()}_OUTPUT_DIR={dest_path}')

        build_temp = self._get_temp_dir(src_dir)
        if self.clean_build:
            remove_tree(build_temp)
        if not os.path.exists(build_temp):
            os.makedirs(build_temp)

        distutils.log.info(f' Configuring from {src_dir} '.center(80, '-'))
        distutils.log.info(f'CMake command: {" ".join(self.cmake_cmd + [src_dir] + args)}')
        distutils.log.info(f'   cwd: {build_temp}')
        try:
            subprocess.check_call(self.cmake_cmd + [src_dir] + args, cwd=build_temp, env=env)
        except ext_errors as err:
            raise BuildFailed() from err
        finally:
            distutils.log.info(f' End configuring from {src_dir} '.center(80, '-'))

    def build_extension(self, ext):
        """Build a single C/C++ extension using CMake."""
        distutils.log.info(f' Building {ext.pymod} '.center(80, '-'))
        distutils.log.info(
            'CMake command: {" ".join(self.cmake_cmd + ["--build", ".", "--target", ext.target] + self.build_args)}'
        )
        distutils.log.info(f'   cwd: {self._get_temp_dir(ext.src_dir)}')
        try:
            subprocess.check_call(
                self.cmake_cmd + ['--build', '.', '--target', ext.target] + self.build_args,
                cwd=self._get_temp_dir(ext.src_dir),
            )
        except ext_errors as err:
            if not ext.optional:
                raise BuildFailed() from err
            distutils.log.info(f'Failed to compile optional extension {ext.target} (not an error)')
        finally:
            distutils.log.info(f' End building {ext.pymod} '.center(80, '-'))

    def copy_extensions_to_source(self):
        """Copy the extensions."""
        # pylint: disable=protected-access

        build_py = self.get_finalized_command('build_py')
        for ext in self.extensions:
            fullname = self.get_ext_fullname(ext.name)
            filename = self.get_ext_filename(fullname)
            modpath = fullname.split('.')
            package = '.'.join(modpath[:-1])
            package_dir = build_py.get_package_dir(package)
            dest_filename = os.path.join(package_dir, os.path.basename(filename))
            src_filename = os.path.join(self.build_lib, filename)

            # Always copy, even if source is older than destination, to ensure
            # that the right extensions for the current Python/platform are
            # used.
            if os.path.exists(src_filename) or not ext.optional:
                copy_file(src_filename, dest_filename, verbose=self.verbose, dry_run=self.dry_run)
                if ext._needs_stub:
                    self.write_stub(package_dir or os.curdir, ext, True)

    def get_outputs(self):
        """
        Get the list of files generated during a build.

        Mainly defined to properly handle optional extensions.
        """
        self.check_extensions_list(self.extensions)
        outputs = []
        for ext in self.extensions:
            if os.path.exists(self.get_ext_fullpath(ext.name)) or not ext.optional:
                outputs.append(self.get_ext_fullpath(ext.name))
        return outputs

    def _get_temp_dir(self, src_dir):
        return os.path.join(self.build_temp, os.path.basename(src_dir))


# ==============================================================================


class Clean(clean):
    """Custom clean command."""

    def run(self):
        """Run the clean command."""
        # Execute the classic clean command
        clean.run(self)
        import glob  # pylint: disable=import-outside-toplevel

        pkg_name = self.distribution.get_name().replace('-', '_')
        info = glob.glob(f'{pkg_name}.egg-info')
        if info:
            remove_tree(info[0])


# ==============================================================================


class GenerateRequirementFile(Command):
    """A custom command to list the dependencies of the current."""

    description = 'List the dependencies of the current package'
    user_options = [
        ('include-all-extras', None, 'Include all "extras_require" into the list'),
        ('include-extras=', None, 'Include some of extras_requires into the list (comma separated)'),
    ]

    boolean_options = ['include-all-extras']

    def initialize_options(self):
        """Initialize this command's options."""
        self.include_extras = None
        self.include_all_extras = None
        self.extra_pkgs = []

    def finalize_options(self):
        """Finalize this command's options."""
        if self.include_extras:
            include_extras = self.include_extras.split(',')
        else:
            include_extras = []

        try:
            for name, pkgs in self.distribution.extras_require.items():
                if self.include_all_extras or name in include_extras:
                    self.extra_pkgs.extend(pkgs)

        except TypeError:  # Mostly for old setuptools (< 30.x)
            for name, pkgs in self.distribution.command_options['options.extras_require'].items():
                if self.include_all_extras or name in include_extras:
                    self.extra_pkgs.extend(pkgs)

    def run(self):
        """Execute this command."""
        with fdopen('requirements.txt', 'w') as req_file:
            try:
                for pkg in self.distribution.install_requires:
                    req_file.write(f'{pkg}\n')
            except TypeError:  # Mostly for old setuptools (< 30.x)
                for pkg in self.distribution.command_options['options']['install_requires']:
                    req_file.write(f'{pkg}\n')
            req_file.write('\n')
            for pkg in self.extra_pkgs:
                req_file.write(f'{pkg}\n')


# ==============================================================================


ext_modules = [
    CMakeExtension(pymod='mindquantum.libQuEST', target='QuEST', optional=True),
    CMakeExtension(pymod='mindquantum.mqbackend'),
]


if __name__ == '__main__':
    remove_tree(os.path.join(cur_dir, 'output'))
    cmake_extra_options.extend(get_extra_cmake_options())
    setuptools.setup(
        # use_scm_version={'local_scheme': 'no-local-version'},
        # setup_requires=['setuptools_scm'],
        cmdclass={
            'build_ext': CMakeBuildExt,
            'clean': Clean,
            'gen_reqfile': GenerateRequirementFile,
        },
        ext_modules=ext_modules,
    )
