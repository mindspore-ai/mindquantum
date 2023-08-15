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
"""Safe open() function."""

import contextlib
import os


@contextlib.contextmanager
def fdopen(fname, mode, perms=0o600, encoding=None):  # pragma: no cover
    """
    Context manager for opening files with correct permissions.

    Args:
        fname (str): Path to file to open for reading/writing
        mode (str): Mode in which the file is opened (see help for builtin `open()`)
        perms (int): Permission mask (see help for `os.open()`)
        encoding (str): The name of encoding used to decode or encode the file.
    """
    if 'r' in mode:
        flags = os.O_RDONLY
    elif 'w' in mode:
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    elif 'a' in mode:
        flags = os.O_WRONLY | os.O_CREAT
    else:
        raise RuntimeError(f'Unsupported mode: {mode}')

    file_object = open(os.open(fname, flags, perms), mode=mode, encoding=encoding)  # noqa: SCS109

    try:
        yield file_object
    finally:
        file_object.close()
