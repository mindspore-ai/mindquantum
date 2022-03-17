# -*- coding: utf-8 -*-
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
import sys

import setuptools.build_meta
from utils import get_cmake_command

build_sdist = setuptools.build_meta.build_sdist
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
    name = os.path.basename(fname)
    sha256_hash = hashlib.sha256()
    with open(fname, 'rb') as wheel_file:
        # Read and update hash string value in blocks of 1M
        for byte_block in iter(lambda: wheel_file.read(1 << 20), b""):
            sha256_hash.update(byte_block)

    with open(f'{fname}.sha256', 'w') as digest_file:
        digest_file.write(f'{sha256_hash.hexdigest()} {name}\n')


def build_sdist(sdist_directory, config_settings=None):
    """Build a source distribution."""
    name = setuptools.build_meta.build_sdist(sdist_directory=sdist_directory, config_settings=config_settings)
    generate_digest_file(os.path.join(sdist_directory, name))
    return name


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    """Build a wheel from this project."""
    if platform.system() == 'Darwin' and (
        '-p' not in config_settings['--global-option'] and '--plat-name' not in config_settings['--global-option']
    ):
        os.environ.setdefault('MACOSX_DEPLOYMENT_TARGET', '10.13')
        os.environ.setdefault('_PYTHON_HOST_PLATFORM', f'macosx-10.13-{platform.machine()}')

    name = setuptools.build_meta.build_wheel(
        wheel_directory=wheel_directory, config_settings=config_settings, metadata_directory=metadata_directory
    )

    name_full = os.path.join(wheel_directory, name)

    # ==========================================================================
    # Delocate the wheel if requested

    delocate_wheel = int(os.environ.get('MQ_DELOCATE_WHEEL', False))
    if delocate_wheel and platform.system() == 'Linux':
        try:
            import auditwheel.main  # pylint: disable=import-outside-toplevel

            plat = os.environ.get('MQ_DELOCATE_WHEEL_PLAT', f'linux_{platform.machine()}')
            if plat.lower() == 'auto' or not plat:
                if sys.version_info.major == 3:
                    plat = f'manylinux2014_{platform.machine()}'
                    if sys.version_info.minor <= 6:
                        plat = f'manylinux1_{platform.machine()}'
                    elif sys.version_info.minor <= 9:
                        plat = f'manylinux2010_{platform.machine()}'

            argv = sys.argv
            sys.argv = [
                'auditwheel',
                'repair',
                '--plat',
                plat,
                '-w',
                wheel_directory,
                name_full,
            ]

            logging.info('Calling %s', ' '.join(sys.argv))

            auditwheel.main.main()

            sys.argv = argv
        except ImportError as err:
            raise RuntimeError('Cannot delocate wheel on Linux without the `auditwheel` package installed!') from err
    elif delocate_wheel and platform.system() == 'Darwin':
        try:
            from delocate.cmd import (  # pylint: disable=import-outside-toplevel
                delocate_wheel,
            )

            argv = sys.argv
            sys.argv = [
                'delocate-wheel',
                '-v',
                f'--require-archs={platform.machine()}',
                name_full,
            ]

            logging.info('Calling %s', ' '.join(sys.argv))
            delocate_wheel.main()

            sys.argv = argv
        except ImportError as err:
            raise RuntimeError('Cannot delocate wheel on MacOS without the `delocate` package installed!') from err
    elif delocate_wheel:
        raise RuntimeError(f'Do not know how to delocate wheels on {platform.system()}')

    # ==========================================================================
    # Calculate the SHA256 of the wheel

    generate_digest_file(name_full)

    return name
