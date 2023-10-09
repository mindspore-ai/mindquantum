#   Copyright 2022 <Huawei Technologies Co., Ltd>
#   Copyright 2017 ProjectQ-Framework (www.projectq.ch)
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

import contextlib
import errno
import logging
import os
import shutil
import stat
import subprocess
import sys
from operator import itemgetter
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'mindquantum' / 'utils'))
from fdopen import (  # noqa: E402 pylint: disable=wrong-import-position,import-error
    fdopen,
)

# ==============================================================================


@contextlib.contextmanager
def modified_environ(*remove, **update):
    """
    Temporarily updates the ``os.environ`` dictionary in-place.

    The `os.environ` dictionary is updated in-place so that the modification is sure to work in all situations.

    Args:
        remove (List[str]): Environment variables to remove.
        update (Dict[str, str]): Dictionary of environment variables and values to add/update.
    """
    env = os.environ

    def convert_values(value):
        if value is None:
            return ''
        if isinstance(value, str):
            return value
        if isinstance(value, bool):
            return f'{int(value)}'
        return str(value)

    update = {k: convert_values(v) for k, v in update.items()} or {}
    remove = remove or []

    stomped = (set(update.keys()) | set(remove)) & set(env.keys())
    update_after = {k: env[k] for k in stomped}
    remove_after = frozenset(k for k in update if k not in env)

    # pylint: disable=expression-not-assigned
    try:
        env.update(update)
        [env.pop(k, None) for k in remove]
        yield
    finally:
        env.update(update_after)
        [env.pop(k) for k in remove_after]


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
        if os.environ.get('PEP517_BUILD_BACKEND', ''):
            # We are being called from `python3 -m build` -> we actually need to ignore the VIRTUAL_ENV variable in
            # order to get the proper path to the Python executable.
            logging.info('Detected PEP517 build frontend')
            raise KeyError('')
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


def get_executable_in_path(exec_name):
    """
    Retrieve the path to an executable.

    First lookup inside the PATH and then within a virtualenv.
    """
    try:
        logging.info('trying to locate %s inside the PATH', exec_name)
        cmd = shutil.which(exec_name)
        if cmd is not None:
            with fdopen(os.devnull, 'w') as devnull:
                subprocess.check_call([cmd, '--version'], stdout=devnull, stderr=devnull)
            logging.info('  -> found at %s', cmd)
            return cmd
    except (OSError, subprocess.CalledProcessError):
        pass
    logging.info('  -> not found in PATH')

    # Executable not in PATH, should have installed the Python CMake module
    # -> try to find out where it is
    return get_executable(exec_name)


# ==============================================================================

try:
    import tomllib

    def parse_toml(filename):
        """Parse a TOML file."""
        with fdopen(str(filename), 'rb') as toml_file:
            return tomllib.load(toml_file)

except ImportError:
    try:
        import toml

        def parse_toml(filename):
            """Parse a TOML file."""
            return toml.load(filename)

    except ImportError:

        try:
            import tomli

            def parse_toml(filename):
                """Parse a TOML file."""
                with fdopen(str(filename), "rb") as toml_file:
                    return tomli.load(toml_file)

        except ImportError:

            def _find_toml_section_end(lines, start):
                """Find the index of the start of the next section."""
                return (
                    next(filter(itemgetter(1), enumerate(line.startswith('[') for line in lines[start + 1 :])))[0]
                    + start
                    + 1
                )

            def _parse_list(lines):
                """Parse a TOML list into a Python list."""
                # NB: This function expects the TOML list to be formatted like so (ignoring leading and trailing
                #     spaces):
                #     name = [
                #          '...',
                #     ]
                #     Any other format is not supported.
                name = None
                elements = []

                for idx, line in enumerate(lines):
                    if name is None and not line.startswith("'"):
                        name = line.split('=')[0].strip()
                        continue
                    if line.startswith("]"):
                        return (name, elements, idx + 1)
                    elements.append(line.rstrip(',').strip("'").strip('"'))

                raise RuntimeError(f'Failed to locate closing "]" for {name}')

            def _parse_string_value(data, key, line):
                if line.startswith(key):
                    data[key] = line.split('=')[1].strip().strip("'")
                    return True
                return False

            def parse_toml(filename):
                """Very simple parser routine for pyproject.toml."""
                result = {'project': {'optional-dependencies': {}}, 'build-sytem': {}}
                with fdopen(filename, mode='r') as toml_file:
                    lines = [line.strip() for line in toml_file.readlines()]
                lines = [line for line in lines if line and not line.startswith('#')]

                # ----------------------

                start = lines.index('[build-system]')
                data = lines[start : _find_toml_section_end(lines, start)]
                idx = 0
                data_length = len(data)
                while idx < data_length:
                    line = data[idx]
                    shift = 1
                    if line.startswith('requires'):
                        (name, pkgs, shift) = _parse_list(data[idx:])
                        result.setdefault('build-system', {})[name] = pkgs
                    idx += shift

                # ----------------------

                start = lines.index('[project]')
                data = lines[start : _find_toml_section_end(lines, start)]
                idx = 0
                data_length = len(data)
                while idx < data_length:
                    line = data[idx]
                    shift = 1
                    if _parse_string_value(result.setdefault('project', {}), 'name', line):
                        pass
                    elif _parse_string_value(result.setdefault('project', {}), 'description', line):
                        pass
                    elif line.startswith('dependencies'):
                        (name, pkgs, shift) = _parse_list(data[idx:])
                        result.setdefault('project', {})[name] = pkgs
                    idx += shift

                # ----------------------

                start = lines.index('[project.optional-dependencies]')
                data = lines[start + 1 : _find_toml_section_end(lines, start)]
                idx = 0
                data_length = len(data)
                while idx < data_length:
                    (opt_name, opt_pkgs, shift) = _parse_list(data[idx:])
                    result.setdefault('project', {}).setdefault('optional-dependencies', {})[opt_name] = opt_pkgs
                    idx += shift
                return result
