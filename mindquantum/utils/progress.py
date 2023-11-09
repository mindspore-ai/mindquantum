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
"""Rich progress module."""
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)


class TwoLoopsProgress(Progress):
    """
    A progress bar of task with two level for loops.

    Args:
        n_outer_loop (int): Total loop size of outer loop.
        n_inner_loop (int): Total loop size of inner loop.
        outer_loop_name (str): The name that will be shown as the title
            of outer loop progress bar. Default: ``"Epoch"``.
        inner_loop_name (str): The name that will be shown as the title
            of inner loop progress bar. Default: ``"Batch"``.

    Examples:
        >>> import time
        >>> from mindquantum.utils import TwoLoopsProgress
        >>> with TwoLoopsProgress(3, 100) as progress:
        >>>     for ep in range(3):
        >>>         for batch in range(100):
        >>>             progress.update_inner_loop(batch)
        >>>             time.sleep(0.01)
        >>>         progress.update_outer_loop(ep)
    """

    def __init__(
        self, n_outer_loop: int, n_inner_loop: int, outer_loop_name: str = 'Epoch', inner_loop_name: str = 'Batch'
    ):
        """Initialize a two level loop task progress bar."""
        super().__init__(
            SpinnerColumn("runner"),
            TextColumn("[progress.description]{task.description}"),
            TextColumn("[{task.completed}/{task.total}]"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            "Time used:",
            TimeElapsedColumn(),
        )
        self.outer_loop_task = self.add_task(description=f"[bold #11aaff]{outer_loop_name}[/]", total=n_outer_loop)
        self.inner_loop_task = self.add_task(description=f"[bold #22ee44]{inner_loop_name}[/]", total=n_inner_loop)
        self.n_outer_loop = n_outer_loop
        self.n_inner_loop = n_inner_loop

    def __enter__(self) -> "TwoLoopsProgress":
        """Enter method."""
        self.start()
        return self

    def update_inner_loop(self, loop_idx: int):
        """
        Update inner loop progress bar.

        Args:
            loop_idx (int): The index of inner loop.
        """
        if loop_idx < self.n_inner_loop:
            self.advance(self.inner_loop_task, advance=1)

    def update_outer_loop(self, loop_idx: int):
        """
        Update outer loop progress bar.

        Args:
            loop_idx (int): The index of outer loop.
        """
        if loop_idx < self.n_outer_loop:
            self.advance(self.outer_loop_task, advance=1)
        if loop_idx < self.n_outer_loop - 1:
            self.reset(self.inner_loop_task)


class SingleLoopProgress(Progress):
    """
    A progress bar of task with single level for loop.

    Args:
        n_loop (int): Total loop size.
        loop_name (str): The name that will be shown as the title
            of loop progress bar. Default: ``"Task"``.

    Examples:
        >>> import time
        >>> from mindquantum.utils import SingleLoopProgress
        >>> with SingleLoopProgress(100) as progress:
        >>>     for batch in range(100):
        >>>         progress.update_loop(batch)
        >>>         time.sleep(0.01)
    """

    def __init__(self, n_loop: int, loop_name: str = 'Task'):
        """Initialize a single level loop task progress bar."""
        super().__init__(
            SpinnerColumn("runner"),
            TextColumn("[progress.description]{task.description}"),
            TextColumn("[{task.completed}/{task.total}]"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            "Time used:",
            TimeElapsedColumn(),
        )
        self.loop_task = self.add_task(description=f"[bold #11aaff]{loop_name}[/]", total=n_loop)
        self.n_loop = n_loop

    def __enter__(self) -> "SingleLoopProgress":
        """Enter method."""
        self.start()
        return self

    def update_loop(self, loop_idx: int):
        """
        Update loop progress bar.

        Args:
            loop_idx (int): The index of loop.
        """
        if loop_idx < self.n_loop:
            self.advance(self.loop_task, advance=1)
