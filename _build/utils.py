#   Copyright 2022 <Huawei Technologies Co., Ltd>
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""Helper functions for building MindQuantum."""

import errno
import logging
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'mindquantum' / 'utils'))
from fdopen import (  # noqa: E402 pylint: disable=wrong-import-position,import-error
    fdopen,
)

# ==============================================================================


def remove_tree(directory):
    """Remove a directory and its subdirectories."""

    def remove_read_only(func, path, exc_info):
        excvalue = exc_info[1]
        if func in (os.rmdir, os.remove) and excvalue.errno == errno.EACCES:
            os.chmod(  # noqa: SCS119 pylint: disable=os-chmod-unsafe-permissions
                path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO
            )
            func(path)
        else:
            raise exc_info[0].with_traceback(exc_info[1], exc_info[2])

    if os.path.exists(directory):
        logging.info('Removing %s (and everything under it)', directory)
        shutil.rmtree(directory, ignore_errors=False, onerror=remove_read_only)


# ==============================================================================


def get_executable(exec_name):
    """Try to locate an executable in a Python virtual environment."""
    try:
        root_path = os.environ['VIRTUAL_ENV']
        python = os.path.basename(sys.executable)
    except KeyError:
        root_path, python = os.path.split(sys.executable)

    exec_name = os.path.basename(exec_name)

    logging.info('trying to locate %s in %s', exec_name, root_path)

    search_paths = [root_path, os.path.join(root_path, 'bin'), os.path.join(root_path, 'Scripts')]

    # First try executing the program directly
    for base_path in search_paths:
        try:
            cmd = os.path.join(base_path, exec_name)
            with fdopen(os.devnull, 'w') as devnull:
                subprocess.check_call([cmd, '--version'], stdout=devnull, stderr=devnull)
        except (OSError, subprocess.CalledProcessError):
            logging.info('  failed with %s', cmd)
        else:
            logging.info('  command found:%s', cmd)
            return cmd

    # That did not work: try calling it through Python
    for base_path in search_paths:
        try:
            cmd = [python, os.path.join(base_path, exec_name)]
            with fdopen(os.devnull, 'w') as devnull:
                subprocess.check_call(cmd + ['--version'], stdout=devnull, stderr=devnull)
        except (OSError, subprocess.CalledProcessError):
            logging.info('  failed with %s', cmd)
        else:
            logging.info('  command found: %s', cmd)
            return cmd

    logging.info('  command *not* found!')
    return None


def get_cmake_command():
    """Retrieve the path to the CMake executable."""
    try:
        cmd = shutil.which('cmake')
        if cmd is not None:
            with fdopen(os.devnull, 'w') as devnull:
                subprocess.check_call([cmd, '--version'], stdout=devnull, stderr=devnull)
            return cmd
    except (OSError, subprocess.CalledProcessError):
        pass

    # CMake not in PATH, should have installed the Python CMake module
    # -> try to find out where it is
    return get_executable('cmake')
