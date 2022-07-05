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
import shutil
import subprocess
from pathlib import Path

import setuptools.build_meta
from utils import fdopen, get_cmake_command  # pylint: disable=import-error
from wheel_filename import parse_wheel_filename

# ==============================================================================


def call_auditwheel(*args, **kwargs):
    """Call auditwheel."""
    args = ['auditwheel', *(str(s) for s in args)]
    logging.info('auditwheel command: %s', ' '.join(args))

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

prepare_metadata_for_build_wheel = setuptools.build_meta.prepare_metadata_for_build_wheel
get_requires_for_build_sdist = setuptools.build_meta.get_requires_for_build_sdist


def get_requires_for_build_wheel(config_settings=None):
    """Identify packages required for building a wheel."""
    requirements = setuptools.build_meta.get_requires_for_build_wheel(config_settings=config_settings)

    if get_cmake_command() is None:
        requirements.append('cmake')

    delocate_wheel = int(os.environ.get('MQ_DELOCATE_WHEEL', False))
    if delocate_wheel:
        if platform.system() == 'Linux':
            requirements.append('auditwheel')
        elif platform.system() == 'Darwin':
            requirements.append('delocate')
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


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    """Build a wheel from this project."""
    if platform.system() == 'Darwin' and (
        '-p' not in config_settings['--global-option'] and '--plat-name' not in config_settings['--global-option']
    ):
        os.environ.setdefault('MACOSX_DEPLOYMENT_TARGET', '10.13')
        os.environ.setdefault('_PYTHON_HOST_PLATFORM', f'macosx-10.13-{platform.machine()}')

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
    delocated_wheel = None
    if delocate_wheel and platform.system() == 'Linux':
        delocated_wheel_directory.mkdir(parents=True, exist_ok=True)
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

        delocated_wheel = get_delocated_wheel_name(name_full, delocated_wheel_directory)
        if delocated_wheel:
            dest_wheel = move_delocated_wheel(delocated_wheel, wheel_directory)
    elif delocate_wheel and platform.system() == 'Darwin':
        delocated_wheel_directory.mkdir(parents=True, exist_ok=True)
        call_delocate_wheel(
            '-v', '-k', '-w', str(delocated_wheel_directory), f'--require-archs={platform.machine()}', name_full
        )
        delocated_wheel = get_delocated_wheel_name(name_full, delocated_wheel_directory)
        if delocated_wheel:
            dest_wheel = move_delocated_wheel(delocated_wheel, wheel_directory)
    elif delocate_wheel:
        logging.warning('Do not know how to delocate wheels on %s', platform.system())

    if not delocated_wheel:
        logging.info('Delocated wheel not found -> using original wheel instead')
        dest_wheel = Path(wheel_directory, name_full.name)
        shutil.move(str(name_full), str(dest_wheel))

    # ==========================================================================
    # Calculate the SHA256 of the wheel

    generate_digest_file(dest_wheel)

    return name
