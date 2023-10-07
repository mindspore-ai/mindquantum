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

# pylint: disable=invalid-overridden-method

"""SVG module."""

import time

import numpy as np

from mindquantum.io.display.circuit_svg_drawer import (
    BaseComponent,
    Line,
    Rect,
    SVGContainer,
    Text,
    super_align,
)


class AnimationSVG(BaseComponent):
    """Animation a svg property."""

    def __init__(self, svg_id, attr, v_from, v_to, dur, spline=None):  # pylint: disable=too-many-arguments
        """Initialize an AnimationSVG object."""
        super().__init__('animate')
        self.id = svg_id  # pylint: disable=invalid-name
        self.attr = attr
        self.v_from = v_from
        self.v_to = v_to
        self.dur = dur
        if spline is None:
            spline = [0.42, 0, 1, 0.8]
        self.spline = spline
        self.prop = {}
        self.prop['xlink:href'] = f'#{self.id}'
        self.prop['attributeName'] = f'{self.attr}'
        self.prop['from'] = f'{self.v_from}'
        self.prop['to'] = f'{self.v_to}'
        self.prop['dur'] = f'{self.dur}s'
        self.prop['calcMode'] = 'spline'
        self.prop['values'] = f'{self.v_from}; {self.v_to}'
        self.prop['keyTimes'] = "0; 1"
        self.prop['keySplines'] = f'{self.spline[0]} {self.spline[1]} {self.spline[2]} {self.spline[3]};'
        self.prop['fill'] = 'freeze'

    @property
    def bottom(self):
        """Getter for bottom."""
        return -np.inf

    @property
    def top(self):
        """Getter for top."""
        return np.inf

    @property
    def left(self):
        """Getter for left."""
        return np.inf

    @property
    def right(self):
        """Getter for right."""
        return -np.inf

    def change_color(self, color):
        """Getter for change of color."""
        return self

    def scale(self, scale):
        """Getter for scale factor."""
        return self

    def shift(self, x, y):
        """Getter for shift factor."""
        return self


class SVGMeasure(SVGContainer):
    """SVG object of measure result."""

    def __init__(self, res, style):
        """Initialize an SVGMeasure object."""
        super().__init__()
        self.style = style
        self.res = res
        self.table = SVGContainer()
        if self.res.shots == 0:
            raise ValueError("shots cannot be zero.")
        self.max_val = max(res.data.values()) / res.shots
        self.max_val = min(self.max_val / 0.9, self.max_val)
        if self.max_val == 0:
            raise ValueError("Error measure result.")
        self.f = (self.style['n_stick'] - 1) / self.max_val * self.style['v_dis']  # pylint: disable=invalid-name
        main_box = SVGContainer()
        text = self.build_title()
        main_box.add(text)
        main_box.add(self.build_box().shift(0, text.bottom + 10))
        background = Rect(0, 0, main_box.right - main_box.left + 20, main_box.bottom - main_box.top + 20)
        background.fill(self.style['background'])
        main_box.shift(10, 10)
        self.add(background)
        self.add(main_box)

    def build_title(self):
        """Build title."""
        shots_text = Text(0, 0, f"Shots:\n {str(self.res.shots)}")
        shots_text.text_anchor("start")
        shots_text.shift(0, (shots_text.bottom - shots_text.top) / 2)
        keys_text = Text(0, 0, f"Keys: {' '.join(self.res.keys[::-1])}")
        keys_text.text_anchor("start")
        keys_text.shift(0, (shots_text.bottom - shots_text.top) / 2 + shots_text.bottom)
        text = SVGContainer()
        text.add(shots_text)
        text.add(keys_text)
        return text

    def build_box(self):  # pylint: disable=too-many-locals,too-many-statements
        """Build box."""
        animations = SVGContainer()
        max_bar_animations = SVGContainer()
        sampling_animations = SVGContainer()
        stick_len = self.style['stick_len']
        h_axis = SVGContainer()
        val_axis = Line(0, self.style['n_stick'] * self.style["v_dis"], stick_len, stick_len)
        val_axis.stroke(self.style['table_box_line_stroke'])
        val_axis.stroke_width(self.style['table_box_line_width'])
        h_axis.add(val_axis)
        box_h = self.style['bar_dis'] * (len(self.res.data) + 0.5)
        for i, num in enumerate(np.linspace(0, self.max_val, self.style['n_stick'])):
            text = Text(
                i * self.style['v_dis'] + self.style['table_box_line_width'] * 2,
                self.style['stick_len'] - self.style['table_box_line_width'] * 2,
                f'{round(num, 3)}',
            )
            text.dominant_baseline('bottom')
            text.text_anchor('start')
            text.font_size(self.style['label_fontsize'])
            text.fill(self.style['label_fontcolor'])
            line = Line(0, 0, 0, box_h)
            line.shift(i * self.style['v_dis'], self.style['stick_len'])
            line.stroke(self.style['vline_stroke'])
            line.stroke_width(self.style['vline_width'])
            if i == 0:
                line.stroke(self.style['table_box_line_stroke'])
            stick = Line(i * self.style['v_dis'], i * self.style['v_dis'], 0, self.style['stick_len'])
            stick.stroke(self.style['table_box_line_stroke'])
            stick.stroke_width(self.style['table_box_line_width'])
            h_axis.add(stick)
            h_axis.add(text)
            h_axis.add(line)
        max_bar = max(self.res.data.values())
        for idx, (string, n) in enumerate(self.res.data.items()):
            stick = Text(0, 0, string)
            stick.fill(self.style['label_fontcolor'])
            stick.font_size(self.style['label_fontsize'])
            stick.text_anchor('end')
            stick.shift(-self.style['vline_width'] * 2 - self.style['stick_len'], (idx + 1) * self.style['bar_dis'])
            line = Line(-self.style['stick_len'], 0, 0, 0)
            line.stroke(self.style['table_box_line_stroke'])
            line.stroke_width(self.style['table_box_line_width'])
            line.shift(0, (idx + 1) * self.style['bar_dis'])
            h_axis.add(stick)
            h_axis.add(line)
            bar = Rect(0, 0, self.f * n / self.res.shots, self.style['rec_h'])  # pylint: disable=disallowed-name
            bar.shift(0, (idx + 1) * self.style['bar_dis'] - self.style['rec_h'] / 2)
            h_axis.add(bar)
            bar.svg_id(f"bar_{idx}_{time.time_ns()}")
            if idx % 2:
                bar.fill(self.style['first_color'])
            else:
                bar.fill(self.style['second_color'])
            anim = AnimationSVG(bar.get('id'), 'width', 0, bar.get('width'), self.style['anim_time'])
            animations.add(anim)
            if n == max_bar:
                anim = AnimationSVG(
                    bar.get('id'), 'fill', bar.get('fill'), self.style['max_color'], self.style['anim_time'] / 2
                )
                anim.prop['begin'] = f"{self.style['anim_time']}s"
                max_bar_animations.add(anim)
            text = Text(0, 0, str(n))
            text.text_anchor('start')
            text.shift(bar.right + 10, (idx + 1) * self.style['bar_dis'])
            text.fill(self.style['label_fontcolor'])
            text.svg_id(f"bar_text_{idx}_{time.time_ns()}")
            text.fill_opacity("0")
            anim = AnimationSVG(text.get('id'), 'fill-opacity', 0, 1, self.style['anim_time'] / 2)
            anim.prop['begin'] = f"{self.style['anim_time']}s"
            sampling_animations.add(anim)
            h_axis.add(text)

        h_axis.shift(-h_axis.left + 5, -h_axis.top)
        h_axis.add(animations)
        h_axis.add(max_bar_animations)
        h_axis.add(sampling_animations)
        text = Text(0, 0, "probability")
        super_align(h_axis, text, 'middle', 'middle', 'v')
        super_align(h_axis, text, 'top', 'bottom')
        h_axis.add(text)
        return h_axis
