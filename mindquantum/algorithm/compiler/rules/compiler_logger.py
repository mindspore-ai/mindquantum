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
"""Log module for compiler."""


class CompileLog:
    HEAD_BLOCK = 0

    @staticmethod
    def IncreaceHeadBlock():
        CompileLog.HEAD_BLOCK += 2

    @staticmethod
    def DecreaseHeadBlock():
        CompileLog.HEAD_BLOCK -= 2

    @staticmethod
    def W(msg: str):
        return f"\033[1;37m{msg}\033[00m"

    @staticmethod
    def K(msg: str):
        return f"\033[1;30m{msg}\033[00m"

    @staticmethod
    def R(msg: str):
        return f"\033[1;31m{msg}\033[00m"

    @staticmethod
    def G(msg: str):
        return f"\033[1;32m{msg}\033[00m"

    @staticmethod
    def Y(msg: str):
        return f"\033[1;33m{msg}\033[00m"

    @staticmethod
    def B(msg: str):
        return f"\033[1;34m{msg}\033[00m"

    @staticmethod
    def P(msg: str):
        return f"\033[1;35m{msg}\033[00m"

    @staticmethod
    def C(msg: str):
        return f"\033[1;36m{msg}\033[00m"

    @staticmethod
    def R1(msg: str):
        return CompileLog.R(msg)

    @staticmethod
    def R2(msg: str):
        return CompileLog.Y(msg)

    @staticmethod
    def ShowState(state):
        return '[' + ', '.join(str(i) if not i else CompileLog.P(i) for i in state) + ']'

    @staticmethod
    def _level1(msg: str):
        print(' ' * CompileLog.HEAD_BLOCK + f"- {msg}")

    @staticmethod
    def _level2(msg: str):
        print(' ' * CompileLog.HEAD_BLOCK + f"{msg}")

    @staticmethod
    def _level_null(msg: str):
        print(' ' * CompileLog.HEAD_BLOCK + f"{msg}")

    @staticmethod
    def log(msg: str, log_level: int, filter_level: int):
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

class LogIndentation:
    def __enter__(self):
        CompileLog.IncreaceHeadBlock()
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        CompileLog.DecreaseHeadBlock()
        return True
