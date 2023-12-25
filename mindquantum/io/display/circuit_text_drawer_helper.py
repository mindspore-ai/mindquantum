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
"""Helper method for drawing circuit in text style."""
# pylint: disable=too-many-lines,invalid-name,too-few-public-methods,too-many-arguments
import sys
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from rich.console import Console

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal  # pragma: no cover

BOX_TABLE = {
    (0, 0, 1, 1): "┌",
    (0, 0, 1, 2): "┎",
    (0, 0, 1, 3): "╓",
    (0, 0, 2, 1): "┍",
    (0, 0, 2, 2): "┏",
    (0, 0, 3, 1): "╒",
    (0, 0, 3, 3): "╔",
    (1, 0, 1, 0): "─",
    (2, 0, 2, 0): "━",
    (3, 0, 3, 0): "═",
    (1, 0, 0, 1): "┐",
    (1, 0, 0, 2): "┒",
    (1, 0, 0, 3): "╖",
    (2, 0, 0, 1): "┑",
    (2, 0, 0, 2): "┓",
    (3, 0, 0, 1): "╕",
    (3, 0, 0, 3): "╗",
    (0, 1, 0, 1): "│",
    (0, 2, 0, 2): "┃",
    (0, 3, 0, 3): "║",
    (1, 1, 0, 0): "┘",
    (1, 2, 0, 0): "┚",
    (1, 3, 0, 0): "╜",
    (2, 1, 0, 0): "┙",
    (2, 2, 0, 0): "┛",
    (3, 1, 0, 0): "╛",
    (3, 3, 0, 0): "╝",
    (0, 1, 1, 0): "└",
    (0, 1, 2, 0): "┕",
    (0, 1, 3, 0): "╘",
    (0, 2, 1, 0): "┖",
    (0, 2, 2, 0): "┗",
    (0, 3, 1, 0): "╙",
    (0, 3, 3, 0): "╚",
    (1, 1, 0, 1): "┤",
    (1, 1, 0, 2): "┧",
    (1, 2, 0, 1): "┦",
    (1, 2, 0, 2): "┨",
    (1, 3, 0, 3): "╢",
    (2, 1, 0, 1): "┥",
    (2, 1, 0, 2): "┪",
    (2, 2, 0, 1): "┩",
    (2, 2, 0, 2): "┫",
    (3, 1, 0, 1): "╡",
    (3, 3, 0, 3): "╣",
    (1, 1, 1, 0): "┴",
    (1, 1, 2, 0): "┶",
    (1, 2, 1, 0): "┸",
    (1, 2, 2, 0): "┺",
    (1, 3, 1, 0): "╨",
    (2, 1, 1, 0): "┵",
    (2, 1, 2, 0): "┷",
    (2, 2, 1, 0): "┹",
    (2, 2, 2, 0): "┻",
    (3, 1, 3, 0): "╧",
    (3, 3, 3, 0): "╩",
    (0, 1, 1, 1): "├",
    (0, 1, 1, 2): "┟",
    (0, 1, 2, 1): "┝",
    (0, 1, 2, 2): "┢",
    (0, 1, 3, 1): "╞",
    (0, 2, 1, 1): "┞",
    (0, 2, 1, 2): "┠",
    (0, 2, 2, 1): "┡",
    (0, 2, 2, 2): "┣",
    (0, 3, 1, 3): "╟",
    (0, 3, 3, 3): "╠",
    (1, 0, 1, 1): "┬",
    (1, 0, 1, 2): "┰",
    (1, 0, 1, 3): "╥",
    (1, 0, 2, 1): "┮",
    (1, 0, 2, 2): "┲",
    (2, 0, 1, 1): "┭",
    (2, 0, 1, 2): "┱",
    (2, 0, 2, 1): "┯",
    (2, 0, 2, 2): "┳",
    (3, 0, 3, 1): "╤",
    (3, 0, 3, 3): "╦",
    (1, 1, 1, 1): "┼",
    (1, 1, 1, 2): "╁",
    (1, 1, 2, 1): "┾",
    (1, 1, 2, 2): "╆",
    (1, 2, 1, 1): "╀",
    (1, 2, 1, 2): "╂",
    (1, 2, 2, 1): "╄",
    (1, 2, 2, 2): "╊",
    (1, 3, 1, 3): "╫",
    (2, 1, 1, 1): "┽",
    (2, 1, 1, 2): "╅",
    (2, 1, 2, 1): "┿",
    (2, 1, 2, 2): "╈",
    (2, 2, 1, 1): "╃",
    (2, 2, 1, 2): "╉",
    (2, 2, 2, 1): "╇",
    (2, 2, 2, 2): "╋",
    (3, 1, 3, 1): "╪",
    (3, 3, 3, 3): "╬",
}

REV_BOX_TABLE = {j: i for i, j in BOX_TABLE.items()}

AVAILABLETABLE = Literal[
    "┌",
    "┎",
    "╓",
    "┍",
    "┏",
    "╒",
    "╔",
    "─",
    "━",
    "═",
    "┐",
    "┒",
    "╖",
    "┑",
    "┓",
    "╕",
    "╗",
    "│",
    "┃",
    "║",
    "┘",
    "┚",
    "╜",
    "┙",
    "┛",
    "╛",
    "╝",
    "└",
    "┕",
    "╘",
    "┖",
    "┗",
    "╙",
    "╚",
    "┤",
    "┧",
    "┦",
    "┨",
    "╢",
    "┥",
    "┪",
    "┩",
    "┫",
    "╡",
    "╣",
    "┴",
    "┶",
    "┸",
    "┺",
    "╨",
    "┵",
    "┷",
    "┹",
    "┻",
    "╧",
    "╩",
    "├",
    "┟",
    "┝",
    "┢",
    "╞",
    "┞",
    "┠",
    "┡",
    "┣",
    "╟",
    "╠",
    "┬",
    "┰",
    "╥",
    "┮",
    "┲",
    "┭",
    "┱",
    "┯",
    "┳",
    "╤",
    "╦",
    "┼",
    "╁",
    "┾",
    "╆",
    "╀",
    "╂",
    "╄",
    "╊",
    "╫",
    "┽",
    "╅",
    "┿",
    "╈",
    "╃",
    "╉",
    "╇",
    "╋",
    "╪",
    "╬",
]


class BoxStyle:
    """
    Box style.

       1
    0┌ ━ ╮2

    7║   ┋3

    6╚ ═ ┙4
       5
    """

    @staticmethod
    def show_box(style):
        """Show the style."""
        s = style
        res = f"{s[0]}{s[1]}{s[2]}\n{s[7]} {s[3]}\n{s[6]}{s[5]}{s[4]}\n"
        print(res)

    # regular
    # ┌─┐
    # │ │
    # └─┘
    regular = ["┌", "─", "┐", "│", "┘", "─", "└", "│"]

    # regular_open_v
    # ┌ ┐
    # │ │
    # └ ┘
    regular_open_v = ["┌", " ", "┐", "│", "┘", " ", "└", "│"]

    # regular_open_h
    # ┌─┐
    #
    # └─┘
    regular_open_h = ["┌", "─", "┐", " ", "┘", "─", "└", " "]

    # regular_heavy
    # ┏━┓
    # ┃ ┃
    # ┗━┛
    regular_heavy = ["┏", "━", "┓", "┃", "┛", "━", "┗", "┃"]

    # regular_heavy_open_h
    # ┏━┓
    #
    # ┗━┛
    regular_heavy_open_h = ["┏", "━", "┓", " ", "┛", "━", "┗", " "]

    # regular_heavy_open_v
    # ┏ ┓
    # ┃ ┃
    # ┗ ┛
    regular_heavy_open_v = ["┏", " ", "┓", "┃", "┛", " ", "┗", "┃"]

    # double
    # ╔═╗
    # ║ ║
    # ╚═╝
    double = ["╔", "═", "╗", "║", "╝", "═", "╚", "║"]

    # double_open_h
    # ╔═╗
    #
    # ╚═╝
    double_open_h = ["╔", "═", "╗", " ", "╝", "═", "╚", " "]

    # double_open_v
    # ╔ ╗
    # ║ ║
    # ╚ ╝
    double_open_v = ["╔", " ", "╗", "║", "╝", " ", "╚", "║"]

    # rounded
    # ╭─╮
    # │ │
    # ╰─╯
    rounded = ["╭", "─", "╮", "│", "╯", "─", "╰", "│"]

    # rounded_open_h
    # ╭─╮
    #
    # ╰─╯
    rounded_open_h = ["╭", "─", "╮", " ", "╯", "─", "╰", " "]

    # rounded_open_v
    # ╭ ╮
    # │ │
    # ╰ ╯
    rounded_open_v = ["╭", " ", "╮", "│", "╯", " ", "╰", "│"]

    # ud_heavy
    # ┍━┑
    # │ │
    # ┕━┙
    ud_heavy = ["┍", "━", "┑", "│", "┙", "━", "┕", "│"]


def combine(s1: str, s2: str, first_main: bool = False, must_zero: List[int] = None) -> AVAILABLETABLE:
    """Combine two table char into one."""
    if must_zero is None:
        must_zero = []
    t1 = REV_BOX_TABLE.get(s1, None)
    t2 = REV_BOX_TABLE.get(s2, None)
    if t1 is None or t2 is None:
        return None
    out = []
    for i, j in zip(t1, t2):
        if i == 0 and j != 0:
            out.append(j)
        elif i != 0 and j == 0:
            out.append(i)
        elif i != 0 and j != 0:
            out.append(i if first_main else j)
        else:
            out.append(0)
    for i in must_zero:
        out[i] = 0
    return BOX_TABLE.get(tuple(out), None)


class Frame:
    """A frame of text monitor."""

    def __init__(self, x, y, data):
        """Construct a frame."""
        self.x, self.y, self.data = x, y, data


def removeprefix(s: str, prefix: str) -> str:
    """Remove the given prefix in string."""
    if s.startswith(prefix):
        return s[len(prefix) :]
    return s


def removesuffix(s: str, suffix: str) -> str:
    """Remove the given suffix in string."""
    if s.endswith(suffix):
        return s[: -len(suffix)]
    return s


class Monitor:
    """Text monitor."""

    def __init__(self, eles: Union[List["BasicObj"], "BasicObj"]):
        """Construct a text monitor."""
        if isinstance(eles, BasicObj):
            eles = [eles]
        self.frames: List[Frame] = []
        for ele in eles:
            f = ele.__create_frame__()
            if isinstance(f, list):
                self.frames.extend(f)
            else:
                self.frames.append(f)
        self.width = max(i.x.max() for i in self.frames) + 1
        self.high = max(i.y.max() for i in self.frames) + 1
        self.canvas = np.full((self.high, self.width), " ", dtype=object)
        for frame in self.frames:
            valid_poi = np.where((frame.x >= 0) * (frame.y >= 0))
            x, y, data = frame.x[valid_poi], frame.y[valid_poi], frame.data[valid_poi]
            for i, j, k in zip(x, y, data):
                self.canvas[j, i] = k

    def get_str(self):
        """Get the whole string."""
        return "\n".join("".join(i) for i in self.canvas)

    def display(self):
        """Display all char in monitor."""
        console = Console()
        canvas = self.get_str()
        console.print(canvas)


class BasicObj(ABC):
    """
    Basic object in text monitor.

          x
     ╔════╤═════
     ║    ┊
    y╟╌╌╌╌·
     ║
    """

    def __init__(self):
        """Construct a basic object."""
        self.post_process = []
        self.__rich_style__ = {}
        self.__rich_property__ = self.register_rich_prop()

    @abstractmethod
    def shift(self, x, y):
        """Shift a position."""

    @abstractmethod
    def left(self):
        """Get the left side of svg."""

    @abstractmethod
    def right(self):
        """Get the right side of svg."""

    @abstractmethod
    def top(self):
        """Get the top side of svg."""

    @abstractmethod
    def bottom(self):
        """Get bottom side of svg."""

    @abstractmethod
    def __create_frame__(self) -> Frame:
        """Create frame data."""

    @abstractmethod
    def register_rich_prop(self) -> List[str]:
        """Register rich property."""

    def disable_rich(self):
        """Disable all rich style."""
        for prop in self.__rich_property__:
            self.__rich_style__[prop] = getattr(self, prop)
            setattr(self, prop, None)
        return self

    def enable_rich(self):
        """Enable all rich style."""
        for prop in self.__rich_property__:
            if getattr(self, prop) is None:
                setattr(self, prop, self.__rich_style__.get(prop, None))
        return self

    def append_post_process(self, fun: Callable[[Frame], Frame]):
        """
        Append a post process function.

        Post process function will applied when generate the frame.
        """
        self.post_process.append(fun)

    def clean_post_process(self):
        """Clean all post process."""
        self.post_process = []

    def __apply_post_process__(self, frame) -> Frame:
        """Apply post process function one by one on the frame."""
        for fun in self.post_process:
            frame = fun(frame)
        return frame

    def display(self):
        """Display this object."""
        Monitor(self).display()

    def get_width(self) -> int:
        """Get the width of the object."""
        return self.right() - self.left() + 1

    def get_high(self) -> int:
        """Get the height of the object."""
        return self.bottom() - self.top() + 1

    def left_top_origin(self) -> "BasicObj":
        """Shift the left top corner to original point."""
        self.shift(-self.left(), -self.top())
        return self


class Container(BasicObj):
    """Container of basic object."""

    def __init__(self, eles: [Union[BasicObj, List[BasicObj]]] = None):
        """Construct a container."""
        if isinstance(eles, BasicObj):
            eles = [eles]
        self.eles: List[BasicObj] = eles if eles is not None else []
        super().__init__()

    def add(self, ele: BasicObj):
        """Add new element to this container."""
        self.eles.append(ele)

    def __create_frame__(self) -> Frame:
        """Create the frame based on the current objects."""
        frames = []
        for ele in self.eles:
            f = ele.__create_frame__()
            if isinstance(f, list):
                frames.extend(f)
            else:
                frames.append(f)
        return frames

    def display(self):
        """Display this container with the help of monitor."""
        Monitor(self.eles).display()

    def shift(self, x: float, y: float) -> "Container":
        """Shift the container with given value."""
        for ele in self.eles:
            ele.shift(x, y)
        return self

    def left(self) -> int:
        """Get the left side position."""
        return min(ele.left() for ele in self.eles)

    def right(self) -> int:
        """Get the right side position."""
        return max(ele.right() for ele in self.eles)

    def top(self) -> int:
        """Get the top position."""
        return min(ele.top() for ele in self.eles)

    def bottom(self) -> int:
        """Get the bottom position."""
        return max(ele.bottom() for ele in self.eles)

    def disable_rich(self):
        """Disable rich style of all elements."""
        for ele in self.eles:
            ele.disable_rich()
        return self

    def enable_rich(self):
        """Enable rich style of all elements."""
        for ele in self.eles:
            ele.enable_rich()
        return self

    def register_rich_prop(self) -> List[str]:
        """Register rich property."""
        return []


class Line(BasicObj):
    """The line object."""

    def __init__(self, char: str, x=0, y=0, length=1, direction=0, line_style=None, factor=1):
        """Construct a line."""
        super().__init__()
        if len(char) != 1:
            raise ValueError(f"char requires a string with length as one, but get {len(char)}")
        self.char = char
        self.x = x
        self.y = y
        self.length = length
        self.direction = direction
        self.line_style = line_style
        self.factor = factor

    def register_rich_prop(self) -> List[str]:
        """Register rich property."""
        return ['line_style']

    def end_point(self) -> Tuple[float, float]:
        """Get the end point coordinate."""
        x = self.x + int(np.cos(self.direction) * self.length)
        y = self.y + int(np.sin(self.direction) * self.factor * self.length)
        return (x, y)

    def left(self) -> int:
        """Get the left side position."""
        return min(self.x, self.end_point()[0])

    def right(self) -> int:
        """Get the right side position."""
        return max(self.x, self.end_point()[0])

    def top(self) -> int:
        """Get the top position."""
        return min(self.y, self.end_point()[1])

    def bottom(self) -> int:
        """Get the bottom position."""
        return max(self.y, self.end_point()[1])

    def shift(self, x, y) -> "Line":
        """Shift the line."""
        self.x += x
        self.y += y
        return self

    def set_length(self, length):
        """Set the length of this line."""
        self.length = length

    def __dye_line__(self, char) -> str:
        """Dye the given char with the same style of this line."""
        if self.line_style is None:
            return char
        return f"[{self.line_style}]{char}[/]"

    def __dedye_line__(self, style_char: str) -> str:
        """Remove the style of styled char."""
        head = f"[{self.line_style}]"
        end = "[/]"
        return removesuffix(removeprefix(style_char, head), end)

    def __create_frame__(self) -> Frame:
        """Create a frame of this line."""
        start = (self.x, self.y)
        end = self.end_point()
        n_point = max(self.right() - self.left() + 1, self.bottom() - self.top() + 1)
        x = np.linspace(start[0], end[0], n_point, dtype=int)
        y = np.linspace(start[1], end[1], n_point, dtype=int)
        char = self.__dye_line__(self.char)
        data = np.full(x.shape, char, dtype=object)
        return self.__apply_post_process__(Frame(x, y, data))


class HLine(Line):
    """Horizontal line."""

    def __init__(
        self,
        x=0,
        y=0,
        length=1,
        line_style=None,
        factor=1,
        thickness: ["normal", "thick", "double"] = "normal",
    ):
        """Construct a horizontal line."""
        char = "─"
        if thickness == "thick":
            char = "━"
        elif thickness == "double":
            char = "═"
        super().__init__(char, x, y, length, 0, line_style, factor)

    def end_point(self) -> Tuple[float, float]:
        """Get the end point of horizontal line."""
        return (self.x + self.length - 1, self.y)


class VLine(Line):
    """Vertical line."""

    def __init__(
        self,
        x=0,
        y=0,
        length=1,
        line_style=None,
        factor=1,
        thickness: Optional[Literal["normal", "thick", "double"]] = "normal",
    ):
        """Construct a vertical line."""
        char = "│"
        if thickness == "thick":
            char = "┃"
        elif thickness == "double":
            char = "║"
        super().__init__(char, x, y, length, np.pi / 2, line_style, factor)

    def end_point(self) -> Tuple[float, float]:
        """Get the end point of vertical line."""
        return (self.x, self.y + self.length - 1)


# pylint: disable=too-many-instance-attributes,dangerous-default-value
class Rect(BasicObj):
    """
    Create rectangle.

              x
     ╔════╤═════
     ║    ┊
    y╟╌╌╌╌·
     ║

    Args:
        x (int): x axis position of left top corner.
        y (int): y axis position of left top corner.
        width (int): width of rectangle.
        high (int): high of rectangle.
        bg (str): Background color. If ``None``, no background. Default: ``None``.
        has_stroke (bool): whether has stroke line. Default: ``True``.
        box_style (List[str]): The box style. This list should have exactly 8 elements.
            Default: ``BoxStyle.regular``.
        box_color (str): The box color. Default: ``None``.
    """

    def __init__(
        self,
        x=0,
        y=0,
        width=3,
        high=3,
        bg=None,
        has_stroke=True,
        box_style=BoxStyle.regular,
        box_color=None,
    ):
        """Construct a rectangle."""
        super().__init__()
        self.high, self.width = high, width
        self.bg = bg
        self.has_stroke = has_stroke
        self.x, self.y = x, y
        self.box_color = box_color
        self.box_style = list(box_style)

    def register_rich_prop(self) -> List[str]:
        """Register rich property."""
        return ['bg', 'box_color']

    def __dye_box__(self, char) -> str:
        """Dye the given char with the style of this rectangle."""
        if self.box_color is None:
            return char
        return f"[{self.box_color}]{char}[/]"

    def __dedye_box__(self, style_char: str) -> str:
        """Remove the style of style char."""
        head = f"[{self.box_color}]"
        end = "[/]"
        return removesuffix(removeprefix(style_char, head), end)

    def __create_frame__(self) -> Frame:
        """Create a frame of this rectangle."""
        x = np.arange(self.x, self.x + self.width, dtype=int)
        y = np.arange(self.y, self.y + self.high, dtype=int)
        x, y = np.meshgrid(x, y)
        f = " " if self.bg is None else f"[{self.bg}]█[/]"
        box_style = self.box_style if self.box_color is None else [self.__dye_box__(i) for i in self.box_style]
        data = np.full(x.shape, f, dtype=object)
        if self.has_stroke:
            data[0] = np.full(x.shape[1], box_style[1])
            data[-1] = np.full(x.shape[1], box_style[5])
            data[:, 0] = np.full(x.shape[0], box_style[7])
            data[:, -1] = np.full(x.shape[0], box_style[3])
            data[0, 0] = box_style[0]
            data[0, -1] = box_style[2]
            data[-1, 0] = box_style[6]
            data[-1, -1] = box_style[4]
        return self.__apply_post_process__(Frame(x.flatten(), y.flatten(), data.flatten()))

    def shift(self, x=0, y=0) -> "Rect":
        """Shift rectangle with given distance."""
        self.x += x
        self.y += y
        return self

    def set_width(self, width) -> "Rect":
        """Set the width of rectangle."""
        self.width = width
        return self

    def set_high(self, high) -> "Rect":
        """Set the height of rectangle."""
        self.high = high
        return self

    def set_poi(self, x, y) -> "Rect":
        """Set the left right corner position."""
        self.x, self.y = x, y
        return self

    def left(self) -> int:
        """Get left position."""
        return self.x

    def right(self) -> int:
        """Get right position."""
        return self.x + self.width - 1

    def top(self) -> int:
        """Get top position."""
        return self.y

    def bottom(self) -> int:
        """Get bottom position."""
        return self.y + self.high - 1


# pylint: disable=too-many-instance-attributes
class Text(BasicObj):
    """A simple Text object."""

    def __init__(
        self,
        string: str,
        x=0,
        y=0,
        width=None,
        bg=None,
        str_style=None,
        align: ["l", "c", "r"] = "c",
    ):
        """Construct a text object."""
        super().__init__()
        self.string = string
        if "\n" in self.string:
            raise ValueError("\\n not allowed in string.")
        self.x = x
        self.y = y
        self.str_style = str_style
        self.width = len(self.string) if width is None else max(len(self.string), width)
        self.high = 1
        self.align = align
        self.bg = bg

    def register_rich_prop(self) -> List[str]:
        """Register rich property."""
        return ['bg', 'str_style']

    def __create_frame__(self) -> Frame:
        """Create a frame of this text object."""
        string = [i if self.str_style is None else f"[{self.str_style}]{i}[/]" for i in self.string]
        l_empty = 0
        r_empty = 0
        empty = self.width - len(self.string)
        if self.align == "l":
            r_empty = empty
        elif self.align == "r":
            l_empty = empty
        else:
            r_empty = empty // 2
            l_empty = empty - r_empty
        bg_string = " " if self.bg is None else f"[{self.bg}]█[/]"
        string = [bg_string] * l_empty + string + [bg_string] * r_empty
        x = np.arange(self.x, self.x + self.width, dtype=int)
        y = np.zeros(self.width, dtype=int) + self.y
        data = np.full(x.shape, "", dtype=object)
        for idx, i in enumerate(string):
            data[idx] = i
        return self.__apply_post_process__(Frame(x.flatten(), y.flatten(), data.flatten()))

    def shift(self, x=0, y=0) -> "Text":
        """Shift the text with given distance."""
        self.x += x
        self.y += y
        return self

    def set_width(self, width) -> "Text":
        """Set the width of text."""
        self.width = max(len(self.string), width)
        return self

    def left(self) -> int:
        """Get left position."""
        return self.x

    def right(self) -> int:
        """Get right position."""
        return self.x + self.width - 1

    def top(self) -> int:
        """Get top position."""
        return self.y

    def bottom(self) -> int:
        """Get bottom position."""
        return self.y


def fix_lines_cross(line1: Line, line2: Line, must_zero=None) -> Callable[[Frame], Frame]:
    """Fix the cross section part of two line."""

    def post_process(frame: Frame) -> Frame:
        """Fix cross section problem."""
        if not isinstance(line1, (HLine, VLine)) or not isinstance(line2, (HLine, VLine)):
            return frame
        if line1.__class__ is line2.__class__:
            return frame
        if isinstance(line1, HLine):
            if line1.left() < line2.left() and line1.right() > line2.right():
                if line1.top() > line2.top() and line1.bottom() < line2.bottom():
                    frame.data[line1.top() - line2.top()] = line2.__dye_line__(
                        combine(line1.char, line2.char, must_zero=must_zero)
                    )
                    return frame
        if isinstance(line1, VLine):
            if line2.left() < line1.left() and line2.right() > line1.right():
                if line2.top() > line1.top() and line2.bottom() < line1.bottom():
                    frame.data[line1.left() - line2.left()] = line2.__dye_line__(
                        combine(line1.char, line2.char, must_zero=must_zero)
                    )
                    return frame
        return frame

    line2.append_post_process(post_process)


def fix_line_rec_cross(rect: Rect, line: Line) -> Callable[[Frame], Frame]:
    """Fix cross section of a line and a rectangle."""

    def post_process(frame: Frame) -> Frame:
        if not rect.has_stroke:
            return frame
        l_left, l_right, l_top, l_bottom = (
            line.left(),
            line.right(),
            line.top(),
            line.bottom(),
        )
        r_left, r_right, r_top, r_bottom = (
            rect.left(),
            rect.right(),
            rect.top(),
            rect.bottom(),
        )

        if l_left != l_right and l_top != l_bottom:
            return frame
        if l_left > r_right or l_right < r_left:
            return frame
        if l_bottom < r_top or l_top > r_bottom:
            return frame

        def reconnect(poi, must_zero=None):
            """Reconnect method."""
            box_old = rect.__dedye_box__(frame.data[poi])
            out = combine(box_old, line.char, first_main=True, must_zero=must_zero)
            if out is not None:
                frame.data[poi] = rect.__dye_box__(out)

        def do(lb1, lb2, lb3, rb1, rb2, rb3, mask, poi):
            """Perform method."""
            if lb1 == lb2 and lb3 < rb1:
                must_zero = []
                if lb1 not in (rb2, rb3):
                    must_zero = [mask]
                reconnect(poi, must_zero)

        do(l_left, l_right, l_top, r_top, r_left, r_right, 3, l_left - r_left)
        do(
            l_left,
            l_right,
            r_bottom,
            l_bottom,
            r_left,
            r_right,
            1,
            l_left - r_right - 1,
        )
        do(
            l_bottom,
            l_top,
            l_left,
            r_left,
            r_top,
            r_bottom,
            2,
            (l_bottom - r_top) * rect.get_width(),
        )
        do(
            l_bottom,
            l_top,
            r_right,
            l_right,
            r_top,
            r_bottom,
            0,
            (l_bottom - r_top + 1) * rect.get_width() - 1,
        )
        return frame

    rect.append_post_process(post_process)


class ObjectEditor:
    """Object editor."""

    @staticmethod
    def to_right(fix: BasicObj, obj: BasicObj):
        """Move obj to right side of fix."""
        obj.shift(fix.right() + 1 - obj.left(), 0)

    @staticmethod
    def to_left(fix: BasicObj, obj: BasicObj):
        """Move obj to left side of fix."""
        obj.shift(fix.left() - 1 - obj.right(), 0)

    @staticmethod
    def to_bottom(fix: BasicObj, obj: BasicObj):
        """Move obj to bottom of fix."""
        obj.shift(0, fix.bottom() + 1 - obj.top())

    @staticmethod
    def to_top(fix: BasicObj, obj: BasicObj):
        """Move obj to top of fix."""
        obj.shift(0, fix.top() - 1 - obj.bottom())

    @staticmethod
    def left_align(fix: BasicObj, obj: BasicObj):
        """Left align obj to fix."""
        obj.shift(fix.left() - obj.left(), 0)

    @staticmethod
    def right_align(fix: BasicObj, obj: BasicObj):
        """Right align obj to fix."""
        obj.shift(fix.right() - obj.right(), 0)

    @staticmethod
    def bottom_align(fix: BasicObj, obj: BasicObj):
        """Bottom align obj to fix."""
        obj.shift(0, fix.bottom() - obj.bottom())

    @staticmethod
    def top_align(fix: BasicObj, obj: BasicObj):
        """Top align obj to fix."""
        obj.shift(0, fix.top() - obj.top())

    @staticmethod
    def v_align(fix: BasicObj, obj: BasicObj):
        """Vertical alignment."""
        obj.shift(fix.left() + fix.get_width() // 2 - obj.left() - obj.get_width() // 2, 0)

    @staticmethod
    def h_align(fix: BasicObj, obj: BasicObj):
        """Horizontal alignment."""
        obj.shift(0, fix.top() + fix.get_high() // 2 - obj.top() - obj.get_high() // 2)

    @staticmethod
    def center_align(fix: BasicObj, obj: BasicObj):
        """Center alignment."""
        ObjectEditor.v_align(fix, obj)
        ObjectEditor.h_align(fix, obj)

    @staticmethod
    def h_expand(objs: List[BasicObj], dist=0):
        """Distribute objects horizontal with dist."""
        if len(objs) < 2:
            return
        for i in range(len(objs) - 1):
            ObjectEditor.to_right(objs[i], objs[i + 1])
            objs[i + 1].shift(dist, 0)

    @staticmethod
    def v_expand(objs: List[BasicObj], dist=0):
        """Distribute objects vertical with dist."""
        if len(objs) < 2:
            return
        for i in range(len(objs) - 1):
            ObjectEditor.to_bottom(objs[i], objs[i + 1])
            objs[i + 1].shift(0, dist)

    @staticmethod
    def batch_ops(objs: List[BasicObj], method: Callable[[BasicObj], BasicObj]):
        """Do the given method with fist element and every other element."""
        if len(objs) < 2:
            return
        for i in objs[1:]:
            method(objs[0], i)
