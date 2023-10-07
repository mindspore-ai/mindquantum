# Copyright 2021 Huawei Technologies Co., Ltd
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
"""MindQuantum custom error exception."""


class DeviceNotSupportedError(Exception):
    """Error for not supported device."""

    def __init__(self, device):
        """Initialize a device not supported error."""
        super().__init__()
        self.msg = f"Device {device} not supported."

    def __str__(self):
        """Get error message."""
        return self.msg


class SimNotAvailableError(Exception):
    """Error for not available simulator."""

    def __init__(self, sim, dtype=None):
        """Initialize a simulator not available error."""
        super().__init__()
        if dtype is None:
            self.msg = f"{sim} not available."
        else:
            self.msg = f"{sim} with data type {dtype} not available."

    def __str__(self):
        """Get error message."""
        return self.msg
