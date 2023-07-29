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
"""Log module for compiler."""
# pylint: disable=invalid-name

import typing


class CompileLog:
    """Compile log."""

    HEAD_BLOCK = 0

    @staticmethod
    def B(msg: str):
        """Display in blue."""
        return f"\033[1;34m{msg}\033[00m"

    @staticmethod
    def C(msg: str):
        """Display in cyan."""
        return f"\033[1;36m{msg}\033[00m"

    @staticmethod
    def DecreaseHeadBlock():
        """Decrease the tag block when display message."""
        CompileLog.HEAD_BLOCK -= 2

    @staticmethod
    def G(msg: str):
        """Display in green."""
        return f"\033[1;32m{msg}\033[00m"

    @staticmethod
    def IncreaseHeadBlock():
        """Increase the tag block when display message."""
        CompileLog.HEAD_BLOCK += 2

    @staticmethod
    def K(msg: str):
        """Display in black."""
        return f"\033[1;30m{msg}\033[00m"

    @staticmethod
    def _level1(msg: str):
        """Display level 1 message."""
        print(' ' * CompileLog.HEAD_BLOCK + f"- {msg}")

    @staticmethod
    def _level2(msg: str):
        """Display level 2 message."""
        print(' ' * CompileLog.HEAD_BLOCK + f"{msg}")

    @staticmethod
    def _level_null(msg: str):
        """Display message without level."""
        print(' ' * CompileLog.HEAD_BLOCK + f"{msg}")

    @staticmethod
    def log(msg: str, log_level: int, filter_level: int):
        """
        Display compiler log message.

        Args:
            msg (str): the log message.
            log_level (int): log level. Could be 0, 1, or 2.
            filter_level (int): disable log message by which filter level.
        """
        if log_level == 0:
            return
        log_dict = {
            1: CompileLog._level1,
            2: CompileLog._level2,
            -1: CompileLog._level_null,
        }
        if log_level not in log_dict:
            log_level = -1
        if log_level > filter_level:
            return
        log_dict[log_level](msg)

    @staticmethod
    def P(msg: str):
        """Display in purple."""
        return f"\033[1;35m{msg}\033[00m"

    @staticmethod
    def R(msg: str):
        """Display in red."""
        return f"\033[1;31m{msg}\033[00m"

    @staticmethod
    def R1(msg: str):
        """Display in red."""
        return CompileLog.R(msg)

    @staticmethod
    def R2(msg: str):
        """Display in yellow."""
        return CompileLog.Y(msg)

    @staticmethod
    def ShowState(state: typing.List[bool]):
        """Show compile result state."""
        return '[' + ', '.join(str(i) if not i else CompileLog.P(i) for i in state) + ']'

    @staticmethod
    def W(msg: str):
        """Display in white."""
        return f"\033[1;37m{msg}\033[00m"

    @staticmethod
    def Y(msg: str):
        """Display in yellow."""
        return f"\033[1;33m{msg}\033[00m"


class LogIndentation:
    """Context for increase and decrease message tab."""

    def __enter__(self):
        """Enter context and increase tab."""
        CompileLog.IncreaseHeadBlock()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit context and decrease tab."""
        CompileLog.DecreaseHeadBlock()
        return True
