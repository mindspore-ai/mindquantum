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

import os
import platform
import sys
import distutils.log

import setuptools.build_meta

build_sdist = setuptools.build_meta.build_sdist
prepare_metadata_for_build_wheel = setuptools.build_meta.prepare_metadata_for_build_wheel
get_requires_for_build_sdist = setuptools.build_meta.get_requires_for_build_sdist


def get_requires_for_build_wheel(config_settings=None):
    """Identify packages required for building a wheel."""
    requirements = setuptools.build_meta.get_requires_for_build_wheel(config_settings=config_settings)

    delocate_wheel = int(os.environ.get('MQ_DELOCATE_WHEEL', False))
    if delocate_wheel:
        if platform.system() == 'Linux':
            requirements.append('auditwheel')
        elif platform.system() == 'Darwin':
            requirements.append('delocate')
    return requirements


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    """Build a wheel from this project."""
    name = setuptools.build_meta.build_wheel(
        wheel_directory=wheel_directory, config_settings=config_settings, metadata_directory=metadata_directory
    )

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
                os.path.join(wheel_directory, name),
            ]

            distutils.log.info('Calling ' + ' '.join(sys.argv))

            auditwheel.main.main()

            sys.argv = argv
        except ImportError as err:
            raise RuntimeError('Cannot delocate wheel on Linux without the `auditwheel` package installed!') from err
    elif delocate_wheel and platform.system() == 'Darwin':
        try:
            from delocate.cmd import delocate_wheel  # pylint: disable=import-outside-toplevel

            argv = sys.argv
            sys.argv = [
                'delocate-wheel',
                '-v',
                '--require-archs={platform.machine()}',
                os.path.join(wheel_directory, name),
            ]

            distutils.log.info('Calling ' + ' '.join(sys.argv))
            delocate_wheel.main()

            sys.argv = argv
        except ImportError as err:
            raise RuntimeError('Cannot delocate wheel on MacOS without the `delocate` package installed!') from err
    elif delocate_wheel:
        raise RuntimeError(f'Do not know how to delocate wheels on {platform.system()}')
    return name
