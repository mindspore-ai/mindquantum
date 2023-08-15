# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Add div for svg to enable horizontal scroll bar."""

import argparse
import contextlib
import json
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


parser = argparse.ArgumentParser()
parser.add_argument('dir', help='dir', type=str, default='.')
args = parser.parse_args()

base_path = os.path.realpath(args.dir)


def find_all_ipynb(current_path):
    """Find all ipynb file."""
    out = []
    for root, _, files in os.walk(current_path):
        for filename in files:
            if filename.endswith(".ipynb"):
                out.append(os.path.join(root, filename))
    return out


all_ipynb = find_all_ipynb(base_path)


def add_div(ipynb_file_name):
    """Add div."""
    with fdopen(ipynb_file_name, 'r') as f_open:
        ipynb_data = json.load(f_open)

    a = "<div class=\"nb-html-output output_area\">"
    b = "</div>"
    modified = False
    if 'cells' not in ipynb_data:
        return
    cells = ipynb_data['cells']
    for cell in cells:
        if 'outputs' not in cell:
            continue
        outputs = cell['outputs']
        for output in outputs:
            if 'data' not in output:
                continue
            data = output['data']
            if 'image/svg+xml' not in data:
                continue
            value = data['image/svg+xml']
            if isinstance(value, list):
                value = ''.join(value)
            if not value.startswith(a):
                value = a + value + b
                data['image/svg+xml'] = value
                modified = True
    if modified:
        with fdopen(ipynb_file_name, 'w') as f_write:
            f_write.writelines(json.dumps(ipynb_data, indent=2))


for f in all_ipynb:
    add_div(f)
