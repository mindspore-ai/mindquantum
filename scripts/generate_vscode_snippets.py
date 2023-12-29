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
"""Generate snippets for vscode."""

import json
import os

from mindquantum.io.display.circuit_text_drawer_helper import removesuffix
from mindquantum.utils import fdopen


def api_snippets(directory, snippets):
    """Generate api snippets."""
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(  # pylint: disable=replace-os-relpath-abspath # noqa: SCS100
                file_path, directory
            )
            en_url = removesuffix(f"https://www.mindspore.cn/mindquantum/docs/en/master/{rel_path}", ".rst") + ".html"
            cn_url = (
                removesuffix(f"https://www.mindspore.cn/mindquantum/docs/zh-CN/master/{rel_path}", ".rst") + ".html"
            )
            api_name = en_url.split('/')[-1].split('.')[-2]
            snippets[f'en {api_name}'] = {
                "prefix": f"url_en_{api_name}",
                "body": en_url,
                "description": f"{api_name} en api",
            }
            snippets[f'cn {api_name}'] = {
                "prefix": f"url_cn_{api_name}",
                "body": cn_url,
                "description": f"{api_name} cn api",
            }


snippets_json = {}
DIRECTORY_PATH = "../docs/api_python"
SNIPPETS_DIR = "../tmp"
SNIPPETS_DIR = os.path.realpath(SNIPPETS_DIR)
DIRECTORY_PATH = os.path.realpath(DIRECTORY_PATH)

snippets_path = os.path.join(SNIPPETS_DIR, 'mindquantum_snippets.code-snippets')

api_snippets(DIRECTORY_PATH, snippets_json)
with fdopen(snippets_path, 'w') as f_write:
    f_write.writelines(json.dumps(snippets_json, indent=2))
print(f'snippets dumped to {snippets_path}')
