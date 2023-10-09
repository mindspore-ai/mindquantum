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

import argparse
import itertools
import logging
import multiprocessing
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import sysconfig
import warnings
from distutils.command.clean import clean  # pylint: disable=deprecated-module
from pathlib import Path

import setuptools
from setuptools.command.build_ext import build_ext
from wheel.bdist_wheel import bdist_wheel

sys.path.append(str(Path(__file__).parent.resolve()))

from _build.utils import (  # pylint: disable=wrong-import-position  # noqa: E402
    fdopen,
    get_executable,
    get_executable_in_path,
    modified_environ,
    parse_toml,
    remove_tree,
)

# ==============================================================================
# Helper variables

on_rtd = os.environ.get('READTHEDOCS') == 'True'
cur_dir = Path(__file__).resolve().parent
ext_errors = (subprocess.CalledProcessError, FileNotFoundError)
cmake_extra_options = []

# ==============================================================================
# Helper functions and classes


def important_msgs(*msgs):
    """Print an important message."""
    print('*' * 75)
    for msg in msgs:
        print(msg)
    print('*' * 75)


# ==============================================================================


def get_python_executable():
    """Retrieve the path to the Python executable."""
    return get_executable(sys.executable)


# ==============================================================================


class BuildFailedError(Exception):
    """Extension raised if the build fails for any reason."""

    def __init__(self):
        """Initialize a BuildFailedError exception."""
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
        self.src_dir = Path().resolve()
        self.pymod = pymod
        self.target = target if target is not None else pymod.split('.')[-1]

        self.lib_filepath = str(Path(*pymod.split('.')))
        super().__init__(pymod, sources=[], optional=optional)


# ------------------------------------------------------------------------------


class CMakeBuildExt(build_ext):  # pylint: disable=too-many-instance-attributes
    """Custom build_ext command class."""

    user_options = build_ext.user_options + [
        ('build-dir=', None, 'Specify a location for the build directory'),
        ('clean-build', None, 'Build in a clean build environment'),
        ('install-light', None, 'Install a "light" version of MindQuantum (ie. no development libraries)'),
        ('jobs=', None, 'Number of concurrent jobs for sub-build processes'),
        ('no-arch-native', None, 'Do not use the -march=native flag when compiling'),
    ]

    boolean_options = build_ext.boolean_options + ['no-arch-native', 'clean-build', 'install-light']

    def initialize_options(self):
        """Initialize all options of this custom command."""
        build_ext.initialize_options(self)
        self.no_arch_native = None
        self.clean_build = None
        self.build_dir = None
        self.install_light = None
        self.jobs = None

    def finalize_options(self):
        """Finalize all options."""
        super().finalize_options()
        # pylint: disable=attribute-defined-outside-init
        self.build_dir = self.build_dir or None
        self.clean_build = self.clean_build or False
        self.fast_bdist_wheel = bool(int(os.getenv('MQ_FAST_BDIST_WHEEL', '0')))
        self.fast_bdist_wheel_dir = os.getenv('MQ_FAST_BDIST_DIR', None)
        self.install_light = self.install_light or False
        self.jobs = self.jobs or multiprocessing.cpu_count()
        self.no_arch_native = self.no_arch_native or False

        if self.fast_bdist_wheel and Path(self.fast_bdist_wheel_dir).exists():
            self.build_dir = self.fast_bdist_wheel_dir
        elif self.fast_bdist_wheel:
            logging.warning('WARN: Disabling fast-build because specified build directory does not exist')
            self.fast_bdist_wheel = False

    def build_extensions(self):
        """Build a C/C++ extension using CMake."""
        # pylint: disable=attribute-defined-outside-init
        if on_rtd:
            important_msgs('skipping CMake build on ReadTheDocs and creating dummy extension packages')
            ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
            for ext in self.extensions:
                dest_path = pathlib.Path(self.get_ext_fullpath(ext.lib_filepath).rstrip(ext_suffix)).with_suffix('.py')
                if not dest_path.exists():
                    logging.info('creating empty file at %s', dest_path)
                    dest_path.write_text('', encoding='utf-8')
            return
        cmake_cmd = get_executable_in_path('cmake')
        if cmake_cmd is None:
            raise RuntimeError('Unable to locate the CMake command!')
        self.cmake_cmd = [cmake_cmd]
        logging.info('using cmake command: %s', ' '.join(self.cmake_cmd))

        if not self.fast_bdist_wheel:
            self.configure_extensions()
        else:
            self.build_args = []

        build_ext.build_extensions(self)

        if self.fast_bdist_wheel:
            self.cmake_install_fast_build()
        elif not self.install_light:
            self.cmake_install()

    def configure_extensions(self):
        """Run a CMake configuration and generation step for one extension."""
        # pylint: disable=attribute-defined-outside-init

        def _src_dir_pred(ext):
            return ext.src_dir

        python_exec = get_python_executable()
        if not python_exec:
            raise RuntimeError('Unable to locate Python executable!')

        pkg_name = self.distribution.get_name()
        if pkg_name == 'UNKNOWN':
            warnings.warn('Unable to determine package name automatically... defaulting to `mindquantum`')
            pkg_name = 'mindquantum'

        cmake_args = [
            '-DPython_EXECUTABLE:FILEPATH=' + python_exec,
            '-DBUILD_TESTING:BOOL=OFF',
            '-DIN_PLACE_BUILD:BOOL=OFF',
            '-DIS_PYTHON_BUILD:BOOL=ON',
            f'-DVERSION_INFO="{self.distribution.get_version()}"',
            f'-DMQ_PYTHON_PACKAGE_NAME:STRING={pkg_name}',
            # NB: make sure that the install path is absolute!
            f'-DCMAKE_INSTALL_PREFIX:FILEPATH={Path(self.build_lib, Path().resolve().name).resolve()}',
        ]

        if self.no_arch_native:
            cmake_args += ['-DUSE_NATIVE_INTRINSICS=OFF']

        cfg = 'Debug' if self.debug else 'Release'
        self.build_args = ['--config', cfg]

        if platform.system() == 'Windows':
            pass
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            self.build_args += [f'-j {self.jobs}', '--']

        cmake_args.extend(cmake_extra_options)

        env = os.environ.copy()

        # This can in principle handle the compilation of extensions outside the main CMake directory (ie. outside the
        # one containing this setup.py file)
        for src_dir, extensions in itertools.groupby(sorted(self.extensions, key=_src_dir_pred), key=_src_dir_pred):
            args = cmake_args.copy()
            for ext in extensions:
                dest_path = Path(self.get_ext_fullpath(ext.lib_filepath)).resolve().parent
                args.append(f'-D{ext.target.upper()}_OUTPUT_DIR={dest_path}')
            self.cmake_configure_build(str(src_dir), args, env)

    def cmake_configure_build(self, src_dir, cmake_args, env):
        """Run a CMake build command for a list of extensions."""
        build_temp = self._get_temp_dir(src_dir)
        if self.clean_build:
            remove_tree(build_temp)
        if not Path(build_temp).exists():
            Path(build_temp).mkdir(parents=True, exist_ok=True)

        build_temp = str(build_temp)

        logging.info(' Configuring from %s '.center(80, '-'), src_dir)
        logging.info('CMake command: %s', ' '.join(self.cmake_cmd + [src_dir] + cmake_args))
        logging.info('   cwd: %s', str(build_temp))
        try:
            subprocess.check_call(self.cmake_cmd + [src_dir] + cmake_args, cwd=build_temp, env=env)
        except ext_errors as err:
            raise BuildFailedError() from err
        finally:
            logging.info(' End configuring from %s '.center(80, '-'), src_dir)

    def build_extension(self, ext):
        """Build a single C/C++ extension using CMake."""
        cwd = self._get_temp_dir(Path(ext.src_dir).resolve().name)
        logging.info(f' Building {ext.pymod} '.center(80, '-'))
        logging.info(
            'CMake command: %s', ' '.join(self.cmake_cmd + ['--build', '.', '--target', ext.target] + self.build_args)
        )
        logging.info('   cwd: %s', cwd)
        try:
            subprocess.check_call(
                self.cmake_cmd + ['--build', '.', '--target', ext.target] + self.build_args,
                cwd=cwd,
            )
            if self.fast_bdist_wheel:
                dest_path = Path(self.get_ext_fullpath(ext.lib_filepath)).resolve()

                for library_path in (
                    cur_dir / Path(ext.lib_filepath).parent / dest_path.name,
                    cur_dir / dest_path.name,
                ):
                    logging.info('[FASTBUILD] trying to locate lib in %s', str(library_path))
                    if library_path.exists():
                        shutil.copyfile(library_path, dest_path)
                        break
                else:
                    raise RuntimeError(f'Unable to locate output file for {ext.name}')
        except ext_errors as err:
            if not ext.optional:
                raise BuildFailedError() from err
            logging.info('Failed to compile optional extension %s (not an error)', ext.pymod)
        finally:
            logging.info(' End building %s '.center(80, '-'), ext.pymod)

    def cmake_install(self):
        """Run the CMake installation step."""
        cwd = self._get_temp_dir(Path().resolve().name)
        logging.info(' Building CMake install target '.center(80, '-'))
        logging.info(
            'CMake command: %s', ' '.join(self.cmake_cmd + ['--build', '.', '--target', 'install'] + self.build_args)
        )
        logging.info('   cwd: %s', cwd)
        try:
            subprocess.check_call(
                self.cmake_cmd + ['--build', '.', '--target', 'install'] + self.build_args,
                cwd=cwd,
            )
        finally:
            logging.info(' End building target install '.center(80, '-'))

    def cmake_install_fast_build(self):
        """Run the CMake installation step for a fast-build."""
        # First save the original installation directory
        install_prefix = None
        with fdopen(str(Path(self.build_dir) / 'CMakeCache.txt'), 'r') as cache_file:
            data = cache_file.readlines()
            cmake_install_prefix = [line.strip() for line in data if line.startswith('CMAKE_INSTALL_PREFIX')]
            if len(cmake_install_prefix) == 1:
                install_prefix = cmake_install_prefix[0].split('=')[1]

        if not install_prefix:
            logging.info('Unable to locate the original installation prefix in %s', self.build_dir)
            logging.info('-> Skipping installation step')
            return

        # ------------------------------

        env = os.environ.copy()

        def _src_dir_pred(ext):
            return ext.src_dir

        # Change the original installation prefix path
        for src_dir, _ in itertools.groupby(sorted(self.extensions, key=_src_dir_pred), key=_src_dir_pred):
            self.cmake_configure_build(
                str(src_dir),
                [f'-DCMAKE_INSTALL_PREFIX:FILEPATH={Path(self.build_lib, Path().resolve().name).resolve()}'],
                env,
            )

        # Perform a normal CMake installation
        self.cmake_install()

        # Restore the original installation prefix path
        for src_dir, _ in itertools.groupby(sorted(self.extensions, key=_src_dir_pred), key=_src_dir_pred):
            self.cmake_configure_build(
                str(src_dir),
                [f'-DCMAKE_INSTALL_PREFIX:FILEPATH={install_prefix}'],
                env,
            )

    def copy_extensions_to_source(self):
        """Copy the extensions."""
        # pylint: disable=protected-access

        build_py = self.get_finalized_command('build_py')
        for ext in self.extensions:
            fullname = self.get_ext_fullname(ext.name)
            filename = Path(self.get_ext_filename(fullname))
            modpath = fullname.split('.')
            package = '.'.join(modpath[:-1])
            package_dir = build_py.get_package_dir(package)
            dest_filename = Path(package_dir, filename.name)
            src_filename = Path(self.build_lib, filename)

            # Always copy, even if source is older than destination, to ensure
            # that the right extensions for the current Python/platform are
            # used.
            if src_filename.exists() or not ext.optional:
                if self.dry_run or self.verbose:
                    logging.info('copy %s -> %s', src_filename, dest_filename)
                if not self.dry_run:
                    shutil.copyfile(src_filename, dest_filename)
                    if ext._needs_stub:
                        self.write_stub(str(package_dir) or os.curdir, ext, True)

    def get_outputs(self):
        """
        Get the list of files generated during a build.

        Mainly defined to properly handle optional extensions.
        """
        outputs = []
        for ext in self.extensions:
            if Path(self.get_ext_fullpath(ext.name)).exists() or not ext.optional:
                outputs.append(self.get_ext_fullpath(ext.name))
        return outputs

    def _get_temp_dir(self, src_dir):
        if self.build_dir:
            return self.build_dir
        return str(Path(self.build_temp, Path(src_dir).name))


# ==============================================================================


class BdistWheel(bdist_wheel):
    """Custom wheel building command."""

    user_options = bdist_wheel.user_options + [
        ('fast-build-dir=', None, 'Specify the location of an existing build directory (defaults to `build`'),
        ('fast-build', None, 'Do a `fast` wheel build (requires a CMake build with IN_PLACE_BUILD set to `ON`'),
    ]

    boolean_options = bdist_wheel.boolean_options + ['fast-build']

    def initialize_options(self):
        """Initialize all options of this custom command."""
        # pylint: disable=attribute-defined-outside-init
        super().initialize_options()
        self.fast_build = None
        self.fast_build_dir = None

    def finalize_options(self):
        """Finalize all options."""
        # pylint: disable=attribute-defined-outside-init
        super().finalize_options()
        self.fast_build = self.fast_build or False
        self.fast_build_dir = self.fast_build_dir or 'build'

    def run(self):
        """Run the bdist_wheel command."""
        if self.fast_build:
            logging.info('doing a fast-build')
            with modified_environ(MQ_FAST_BDIST_WHEEL=True, MQ_FAST_BDIST_DIR=self.fast_build_dir):
                super().run()
        else:
            super().run()


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


class GenerateRequirementFile(setuptools.Command):
    """A custom command to list the dependencies of the current."""

    description = 'List the dependencies of the current package'
    user_options = [
        ('include-all-extras', None, 'Include all extras_require into the list'),
        ('include-extras=', None, 'Include some of extras_requires into the list (comma separated)'),
        ('output=', 'o', 'Include some of extras_requires into the list (comma separated)'),
    ]

    boolean_options = ['include-all-extras']

    def initialize_options(self):
        """Initialize this command's options."""
        # pylint: disable=attribute-defined-outside-init
        self.include_extras = None
        self.include_all_extras = None
        self.output = None
        self.extra_pkgs = []
        self.dependencies = []

    def finalize_options(self):
        """Finalize this command's options."""
        # pylint: disable=attribute-defined-outside-init
        if not self.output:
            self.output = Path(__file__).parent.resolve() / 'requirements.txt'
        else:
            self.output = Path(self.output)

        include_extras = self.include_extras.split(',') if self.include_extras else []
        pyproject_toml = parse_toml(Path(__file__).parent / 'pyproject.toml')

        for name, pkgs in pyproject_toml['project']['optional-dependencies'].items():
            if self.include_all_extras or name in include_extras:
                self.extra_pkgs.extend(pkgs)

        self.dependencies = self.distribution.install_requires
        if not self.dependencies:
            self.dependencies = pyproject_toml['project']['dependencies']

    def run(self):
        """Execute this command."""
        with fdopen(str(self.output), 'w') as req_file:
            for pkg in self.dependencies:
                req_file.write(f'{pkg}\n')
            req_file.write('\n')
            for pkg in self.extra_pkgs:
                req_file.write(f'{pkg}\n')


# ==============================================================================

ext_modules = [
    CMakeExtension(pymod='mindquantum.mqbackend'),
    CMakeExtension(pymod='mindquantum._mq_vector'),
    CMakeExtension(pymod='mindquantum._mq_vector_gpu', optional=True),
    CMakeExtension(pymod='mindquantum._mq_matrix'),
    CMakeExtension(pymod='mindquantum._math'),
]


# ==============================================================================


class ArgsCMakeFlag(argparse.Action):
    """Custom argparse action for CMake flags."""

    def __call__(self, parser, namespace, values, option_string=None):
        """Call operator."""
        if isinstance(values, list):
            raise ValueError(f'values = {values} is a list! Only single value are currently supported!')

        cmake_extra_options.append(option_string)
        cmake_extra_options.append(values)
        setattr(namespace, self.dest, True)


class ArgsCMakeDefinition(argparse.Action):
    """Custom argparse action to set boolean CMake variables."""

    def __init__(self, cmake_value, *args, **kwargs):
        """Initialize an ArgsCMakeDefinition object."""
        super().__init__(*args, **kwargs)
        self.cmake_value = bool(cmake_value)

    def __call__(self, parser, namespace, values, option_string=None):
        """Call operator."""
        if isinstance(values, list):
            raise ValueError(f'values = {values} is a list! Only single value are currently supported!')

        if self.cmake_value:
            cmake_extra_options.append(f'-D{values}:BOOL=ON')
        else:
            cmake_extra_options.append(f'-D{values}:BOOL=OFF')


class ArgsCMakeVariable(argparse.Action):
    """Custom argparse action to set string CMake variables."""

    def __call__(self, parser, namespace, values, option_string=None):
        """Call operator."""
        if not isinstance(values, list) or len(values) != 2:
            parser.error(
                f'{option_string} expects 2 arguments, only {len(values)} provided! (make sure nargs is set properly)'
            )
        cmake_extra_options.append(f'-D{values[0]}:STRING={values[1]}')


# ==============================================================================

if __name__ == '__main__':
    remove_tree(Path(cur_dir, 'output'))

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--set', action=ArgsCMakeDefinition, cmake_value=True)
    arg_parser.add_argument('--unset', action=ArgsCMakeDefinition, cmake_value=False)
    arg_parser.add_argument('--var', nargs=2, action=ArgsCMakeVariable)
    arg_parser.add_argument('-A', action=ArgsCMakeFlag)
    arg_parser.add_argument('-G', dest='cmake_generator', action=ArgsCMakeFlag)

    if 'bdist_wheel' in sys.argv:
        sys.argv.extend(arg.strip() for arg in os.environ.get('MQ_CIBW_BUILD_ARGS', '').split(',') if arg)
        parsed_args, unparsed_args = arg_parser.parse_known_args()

        sys.argv = sys.argv[:1] + unparsed_args

        # If no explicit CMake Generator specification, prefer MinGW Makefiles on Windows
        if (not parsed_args.cmake_generator) and (platform.system() == 'Windows'):
            cmake_extra_options += ['-G', 'MinGW Makefiles']

    setuptools.setup(
        cmdclass={
            'bdist_wheel': BdistWheel,
            'build_ext': CMakeBuildExt,
            'clean': Clean,
            'gen_reqfile': GenerateRequirementFile,
        },
        ext_modules=ext_modules,
    )
