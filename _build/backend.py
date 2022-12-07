# Copyright 2022 Huawei Technologies Co., Ltd
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

"""Custom build backend."""

import hashlib
import logging
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

import setuptools.build_meta
from utils import fdopen, get_executable_in_path
from wheel_filename import parse_wheel_filename

# ==============================================================================


def call_auditwheel(*args, **kwargs):
    """Call auditwheel."""
    args = ['auditwheel', *(str(s) for s in args)]
    logging.info('auditwheel command: %s', ' '.join(args))
    logging.info('  location of executable: %s', shutil.which('auditwheel'))

    try:
        subprocess.check_call(args)
    except subprocess.CalledProcessError:
        if not kwargs.get('allow_failure', False):
            raise
        return False
    return True


# ------------------------------------------------------------------------------


def call_delocate_wheel(*args):
    """Call delocate-wheel."""
    args = ['delocate-wheel', *(str(s) for s in args)]
    logging.info('delocate-wheel command: %s', ' '.join(args))
    logging.info('  location of executable: %s', shutil.which('delocate-wheel'))

    subprocess.check_call(args)
    return True


# ------------------------------------------------------------------------------


def call_delvewheel(*args):
    """Call delvewheel."""
    args = ['delvewheel', *(str(s) for s in args)]
    logging.info('delvewheel command: %s', ' '.join(args))
    logging.info('  location of executable: %s', shutil.which('delvewheel'))

    subprocess.check_call(args)
    return True


# ------------------------------------------------------------------------------


def move_delocated_wheel(delocated_wheel, wheel_directory):
    """Move delocated wheel to destination directory."""
    logging.info('Delocated wheel found at: %s', str(delocated_wheel))
    dest_wheel = Path(wheel_directory, delocated_wheel.name)
    if dest_wheel.exists():
        logging.info('Destination wheel %s already exists, deleting', str(dest_wheel))
        dest_wheel.unlink()
    logging.info('Moving delocated wheel into destination directory')
    logging.info('  %s -> %s', str(delocated_wheel), str(dest_wheel))
    shutil.move(str(delocated_wheel), str(dest_wheel))
    return dest_wheel


# ------------------------------------------------------------------------------


def get_delocated_wheel_name(name_full, delocated_wheel_directory):
    """Locate the delocated wheel on Linux."""
    pwf = parse_wheel_filename(str(name_full))
    platform_tag_suffix = '.'.join(pwf.platform_tags) + '.whl'
    basename = name_full.name[: -len(platform_tag_suffix)]
    logging.info('Basename of original wheel: %s', basename)
    logging.info('Platform tag suffix: %s', platform_tag_suffix)
    for new_wheel in delocated_wheel_directory.iterdir():
        if new_wheel.is_file() and new_wheel.match(f'{basename}*.whl'):
            return new_wheel
    logging.warning('Unable to locate delocated wheel: %s', str(name_full))
    return None


# ==============================================================================


def update_library_path_var(ld_path_var, dir_list):
    """Update a XXX_LIBRARY_PATH environment variable."""
    ld_library_path = os.getenv(ld_path_var)
    if ld_library_path:
        dir_list.append(ld_library_path)

    logging.info('Setting %s = %s', ld_path_var, os.pathsep.join(dir_list))
    os.environ[ld_path_var] = os.pathsep.join(dir_list)


# ------------------------------------------------------------------------------


def update_library_path_from_file(ld_path_var, filename):
    """Update XXX_LIBRARY_PATH environment variables based on a path list in a file."""
    if not filename or not Path(filename).exists():
        logging.info('Could not locate %s', filename)
        raise FileNotFoundError(filename)

    logging.info('----------------------------------------')
    logging.info('Reading library paths from: %s', filename)

    paths = []
    with fdopen(filename, 'r') as path_file:
        paths = [Path(line.strip()) for line in path_file.readlines()]

    ld_lib_paths = []
    for paths_root in paths:
        for suffix in ('lib64', 'lib'):
            deps_dir = paths_root / suffix
            if deps_dir.is_dir() and deps_dir.exists():
                logging.info('  prepending path to %s: %s', ld_path_var, deps_dir)
                ld_lib_paths.append(str(deps_dir))
            else:
                logging.info('  directory does not exist: %s', deps_dir)

    update_library_path_var(ld_path_var, ld_lib_paths)
    logging.info('----------------------------------------')


# ------------------------------------------------------------------------------


def update_library_path_from_env(ld_path_var, install_prefix):
    """Update XXX_LIBRARY_PATH environment variables based on some folder path."""
    if not install_prefix:
        logging.info('Could not locate install prefix at "%s"', install_prefix)
        raise FileNotFoundError(install_prefix)

    logging.info('----------------------------------------')
    logging.info('Looking into installation prefix: %s', install_prefix)
    ld_lib_paths = []
    deps_dir_list = tuple(
        deps_dir / 'lib64' for deps_dir in Path(install_prefix).iterdir() if deps_dir.is_dir() and deps_dir.exists()
    )
    deps_dir_list += tuple(
        deps_dir / 'lib' for deps_dir in Path(install_prefix).iterdir() if deps_dir.is_dir() and deps_dir.exists()
    )
    for deps_dir in deps_dir_list:
        if deps_dir.exists() and deps_dir.is_dir():
            logging.info('  prepending path to %s: %s', ld_path_var, deps_dir)
            ld_lib_paths.append(str(deps_dir))

    update_library_path_var(ld_path_var, ld_lib_paths)
    logging.info('----------------------------------------')


# ------------------------------------------------------------------------------


def update_library_path(ld_path_var):
    """Update XXX_LIBRARY_PATH environment variable."""
    logging.info('Updating %s environment variable', ld_path_var)
    mq_lib_paths = os.getenv('MQ_LIB_PATHS', '')
    try:
        update_library_path_from_file(ld_path_var, mq_lib_paths)
        return
    except FileNotFoundError:
        pass

    try:
        prefix = 'build/temp.macosx'
        if platform.system() == 'Darwin' and mq_lib_paths.startswith(prefix):
            re_match = re.match(r'build/temp\.macosx-([^-]+)-(.*)', mq_lib_paths)
            if re_match:
                macos_ver = os.getenv('MACOSX_DEPLOYMENT_TARGET', 'NA')
                suffix = re_match.group(2)
                filename = Path(f'{prefix}-{macos_ver}-{suffix}')
                if filename.exists():
                    update_library_path_from_file(ld_path_var, str(filename))
                    return
    except FileNotFoundError:
        pass

    for install_prefix in (
        'MQLIBS_CACHE_PATH',
        'MSLIBS_CACHE_PATH',
        'MQLIBS_LOCAL_PREFIX_PATH',
        'MSLIBS_LOCAL_PREFIX_PATH',
        'MQ_BUILD_DIR',
    ):
        try:
            update_library_path_from_env(ld_path_var, os.getenv(install_prefix, ''))
            return
        except FileNotFoundError:
            pass


# ==============================================================================

prepare_metadata_for_build_wheel = setuptools.build_meta.prepare_metadata_for_build_wheel
get_requires_for_build_sdist = setuptools.build_meta.get_requires_for_build_sdist


def get_requires_for_build_wheel(config_settings=None):
    """Identify packages required for building a wheel."""
    requirements = setuptools.build_meta.get_requires_for_build_wheel(config_settings=config_settings)

    executable_list = ['cmake']
    if int(os.environ.get('MQ_USE_NINJA', False)):
        executable_list.append('ninja')

    delocate_wheel = int(os.environ.get('MQ_DELOCATE_WHEEL', False))
    if delocate_wheel:
        if platform.system() == 'Linux':
            requirements.append('auditwheel')
            executable_list.append('patchelf')
        elif platform.system() == 'Darwin':
            requirements.append('delocate')
        elif platform.system() == 'Windows':
            requirements.append('delvewheel')

    for exec_name in executable_list:
        if get_executable_in_path(exec_name) is None:
            requirements.append(exec_name)

    return requirements


def generate_digest_file(fname):
    """Generate a SHA256 digest file for the wheels."""
    logging.info('Generate SHA256 digest file for %s', fname)
    name = os.path.basename(fname)
    sha256_hash = hashlib.sha256()
    with fdopen(fname, 'rb') as wheel_file:
        # Read and update hash string value in blocks of 1M
        for byte_block in iter(lambda: wheel_file.read(1 << 20), b""):
            sha256_hash.update(byte_block)

    with fdopen(f'{fname}.sha256', 'w') as digest_file:
        digest_file.write(f'{sha256_hash.hexdigest()} {name}\n')


def build_sdist(sdist_directory, config_settings=None):
    """Build a source distribution."""
    name = setuptools.build_meta.build_sdist(sdist_directory=sdist_directory, config_settings=config_settings)
    generate_digest_file(os.path.join(sdist_directory, name))
    return name


# ==============================================================================


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):  # pylint: disable=R0912,R0914,R0915
    """Build a wheel from this project."""
    if platform.system() == 'Linux':
        logging.info('Running on Linux %s', platform.uname().release)
    elif platform.system() == 'Darwin':
        logging.info('Running on macOS v%s', platform.mac_ver()[0])
        if '-p' not in config_settings['--global-option'] and '--plat-name' not in config_settings['--global-option']:
            macos_ver = '10.15'
            os.environ.setdefault('MACOSX_DEPLOYMENT_TARGET', macos_ver)
            os.environ.setdefault('_PYTHON_HOST_PLATFORM', f'macosx-{macos_ver}-{platform.machine()}')
        logging.info('MACOSX_DEPLOYMENT_TARGET = %s', os.environ.get("MACOSX_DEPLOYMENT_TARGET"))
        logging.info('_PYTHON_HOST_PLATFORM = %s', os.environ.get("_PYTHON_HOST_PLATFORM"))

    temp_wheel_directory = Path(wheel_directory, 'temp')
    temp_wheel_directory.mkdir(parents=True, exist_ok=True)

    name = setuptools.build_meta.build_wheel(
        wheel_directory=temp_wheel_directory,
        config_settings=config_settings,
        metadata_directory=metadata_directory,
    )
    name_full = temp_wheel_directory / name

    # ==========================================================================
    # Delocate the wheel if requested

    delocated_wheel_directory = Path(wheel_directory, 'delocated')
    delocate_wheel = int(os.environ.get('MQ_DELOCATE_WHEEL', False))
    done_delocate = False
    delocated_wheel = None

    if delocate_wheel:
        logging.info('Attempting to delocate the generated Python wheel')
        delocated_wheel_directory.mkdir(parents=True, exist_ok=True)
        ld_lib_var = 'LD_LIBRARY_PATH'
        if platform.system() == 'Darwin':
            ld_lib_var = 'DYLD_LIBRARY_PATH'
        elif platform.system() == 'Windows':
            ld_lib_var = 'MQ_LIB'  # Name of variable intentionally custom
        update_library_path(ld_lib_var)

        if platform.system() == 'Linux':
            done_delocate = True
            plat = os.environ.get('MQ_DELOCATE_WHEEL_PLAT', '')

            call_auditwheel('show', name_full)
            if plat:
                call_auditwheel('repair', '--plat', plat, '-w', delocated_wheel_directory, name_full)
            else:
                logging.info('No platform specified, trying a few from older specifications to more recent')
                for plat in (
                    'manylinux2010_x86_64',  # NB: equivalent to manylinux_2_5_x86_64
                    'manylinux2014_x86_64',  # NB: equivalent to manylinux_2_17_x86_64
                    'manylinux_2_24_x86_64',
                    'manylinux_2_27_x86_64',
                    'manylinux_2_28_x86_64',
                    'manylinux_2_31_x86_64',
                    'linux_x86_64',
                ):
                    logging.info('----------------------------------------')
                    logging.info('Trying to delocate to platform: %s', plat)
                    if call_auditwheel(
                        'repair', '--plat', plat, '-w', delocated_wheel_directory, name_full, allow_failure=True
                    ):
                        break
        elif platform.system() == 'Darwin':
            done_delocate = True
            call_delocate_wheel(
                '--verbose',
                '--check-archs',
                '--dylibs-only',
                '-w',
                str(delocated_wheel_directory),
                f'--require-archs={platform.machine()}',
                name_full,
            )
        elif platform.system() == 'Windows':
            done_delocate = True
            args = []
            ld_libs_paths = os.getenv(ld_lib_var)
            if ld_libs_paths:
                args.extend(('--add-path', ld_libs_paths))

            call_delvewheel('show', *args, name_full)
            call_delvewheel('repair', '-v', '-w', str(delocated_wheel_directory), *args, name_full)

        else:
            logging.warning('Do not know how to delocate wheels on %s', platform.system())
    else:
        logging.info('Not delocating the Python wheel')

    if done_delocate:
        delocated_wheel = get_delocated_wheel_name(name_full, delocated_wheel_directory)
        dest_wheel = move_delocated_wheel(delocated_wheel, wheel_directory)
    else:
        logging.info('Delocated wheel not found -> using original wheel instead')
        dest_wheel = Path(wheel_directory, name_full.name)
        shutil.move(str(name_full), str(dest_wheel))

    # ==========================================================================
    # Calculate the SHA256 of the wheel

    generate_digest_file(dest_wheel)

    return name
