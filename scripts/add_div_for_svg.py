# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http: //www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Add div for svg to enable horizontal scroll bar."""

import argparse
import json
import os

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
    with open(ipynb_file_name) as f_open:
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
        with open(ipynb_file_name, 'w') as f_write:
            f_write.writelines(json.dumps(ipynb_data, indent=2))


for f in all_ipynb:
    add_div(f)
