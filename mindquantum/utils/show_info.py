# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Show module info."""
import platform
import time

import psutil

try:
    import importlib.metadata as importlib_metadata
except ImportError:
    import importlib_metadata


def get_version(module_name: str):
    """Get module version."""
    return importlib_metadata.version(module_name)


# pylint: disable=too-few-public-methods
class InfoTable:
    """Show information table of operation system and given modules."""

    def __init__(self, *modules):
        """Initialize a info table object."""
        self.modules = modules
        self.module_versions = [get_version(i) for i in self.modules]

    def sys_info(self):
        """Show system info."""
        res = ""
        res += f"<tr><td>Python</td><td>{platform.python_version()}</td></tr>"
        res += f"<tr><td>OS</td><td>{platform.system()} {platform.machine()}</td></tr>"
        res += f"<tr><td>Memory</td><td>{round(psutil.virtual_memory().total/1e9, 2)} GB</td></tr>"
        res += f"<tr><td>CPU Max Thread</td><td>{psutil.cpu_count()}</td></tr>"
        res += f"<tr><td>Date</td><td>{time.ctime()}</td></tr>"
        return res

    def _repr_html_(self):
        """Return a table that can be rendered by jupyter notebook."""
        module_table = "\n".join(
            f"<tr><td>{n}</td><td>{v}</td></tr>" for n, v in zip(self.modules, self.module_versions)
        )
        return f"""
<table border="1">
  <tr>
    <th>Software</th>
    <th>Version</th>
  </tr>
{module_table}
<tr>
    <th>System</th>
    <th>Info</th>
</tr>
{self.sys_info()}
</table>
"""
