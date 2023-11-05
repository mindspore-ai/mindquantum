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

# pylint: disable=too-many-lines,invalid-overridden-method

"""SVG module."""

import copy
from abc import ABC, abstractmethod

import numpy as np

from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import (
    BarrierGate,
    CNOTGate,
    Measure,
    NoiseGate,
    ParameterGate,
    SGate,
    SWAPGate,
    TGate,
    XGate,
)
from mindquantum.io.display._config import _DAGGER_MASK
from mindquantum.utils import fdopen
from mindquantum.utils.type_value_check import _check_input_type


class BaseComponent(ABC):
    """BaseComponent."""

    def __init__(self, tag):
        """Initialize a BaseComponent object."""
        self.head = f"<{tag}"
        self.tail = "/>"
        self.prop = {}

    def _prop_to_str(self):
        ele = []
        for k, v in self.prop.items():
            ele.append(f"{k}=\"{v}\"")
        return ele

    def __str__(self):
        """Return a string representation of the object."""
        return self.to_string()

    def __repr__(self):
        """Return a string representation of the object."""
        return self.to_string()

    def svg_id(self, id_name):
        """Set id for this svg component."""
        self.prop['id'] = id_name

    def to_string(self):
        """
        SVG to string.

        Returns:
            str, string of svg.
        """
        return " ".join([self.head] + self._prop_to_str() + [self.tail])

    def get(self, prop):
        """Get property of svg parameters."""
        return self.prop[prop]

    @abstractmethod
    def scale(self, scale):
        """Scale the svg figure."""

    @abstractmethod
    def shift(self, x, y):
        """Shift the svg figure."""

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
    def change_color(self, color):
        """Change color of svg."""


class HasStroke(BaseComponent):
    """A svg component that has stroke related property."""

    def stroke(self, stroke):
        """Set stroke."""
        self.prop['stroke'] = stroke
        return self

    def stroke_width(self, stroke_width):
        """Set stroke width."""
        self.prop['stroke-width'] = stroke_width
        return self

    def scale(self, scale):
        """Change the scale of stroke."""
        if 'stroke-width' in self.prop:
            self.stroke_width(self.get('stroke-width') * scale)
        return self

    def change_color(self, color):
        """Change the color of stroke."""
        self.stroke(color)
        return self

    def stroke_dasharray(self, width_a, width_b):
        """Set stroke dash array."""
        self.prop['stroke-dasharray'] = f"{width_a} {width_b}"
        return self


class HasFill(HasStroke):
    """A svg component that has fill property."""

    def fill(self, fill):
        """Set fill."""
        self.prop['fill'] = fill
        return self

    def fill_opacity(self, opacity):
        """Set fill opacity."""
        self.prop['fill-opacity'] = opacity

    def change_color(self, color):
        """Change the color of filled component."""
        self.fill(color)
        return self

    def get_main_color(self):
        """Get main color."""
        if 'fill-opacity' not in self.prop:
            opacity = 0
        else:
            opacity = self.prop['fill-opacity']
        if opacity != 0:
            return self.prop['fill']
        return self.prop['stroke']


class HasXY(BaseComponent):
    """A svg component that has x and y property."""

    def x(self, x):
        """Change x."""
        self.prop['x'] = x
        return self

    def y(self, y):
        """Change y."""
        self.prop['y'] = y
        return self

    def scale(self, scale):
        """Change the size of x and y."""
        self.x(self.get('x') * scale)
        self.y(self.get('y') * scale)
        return self

    def shift(self, x, y):
        """Shift x and y."""
        self.x(self.get('x') + x)
        self.y(self.get('y') + y)
        return self


class Line(HasStroke):
    """SVG line component."""

    def __init__(self, x1, x2, y1, y2):
        """Initialize a Line object."""
        super().__init__('line')
        self.x1(x1)
        self.x2(x2)
        self.y1(y1)
        self.y2(y2)

    def scale(self, scale):
        """Scale the line."""
        self.x1(self.get('x1') * scale)
        self.y1(self.get('y1') * scale)
        self.x2(self.get('x2') * scale)
        self.y2(self.get('y2') * scale)
        HasStroke.scale(self, scale)
        return self

    def x1(self, x1):
        """Change x1."""
        self.prop['x1'] = x1
        return self

    def x2(self, x2):
        """Change x2."""
        self.prop['x2'] = x2
        return self

    def y1(self, y1):
        """Change y1."""
        self.prop['y1'] = y1
        return self

    def y2(self, y2):
        """Change y2."""
        self.prop['y2'] = y2
        return self

    def shift(self, x, y):
        """Shift line."""
        self.x1(self.get('x1') + x)
        self.y1(self.get('y1') + y)
        self.x2(self.get('x2') + x)
        self.y2(self.get('y2') + y)
        return self

    @property
    def left(self):
        """Left side of line."""
        return self.get('x1')

    @property
    def right(self):
        """Right side of line."""
        return self.get('x2')

    @property
    def top(self):
        """Top side of line."""
        return self.get('y1')

    @property
    def bottom(self):
        """Bottom side of line."""
        return self.get('y2')

    def change_color(self, color):
        """Change color of line."""
        self.stroke(color)
        return self


class Rect(HasFill, HasXY):
    """SVG rectangle component."""

    def __init__(self, x, y, width, height):
        """Initialize a Rect object."""
        super().__init__('rect')
        self.x(x)
        self.y(y)
        self.width(width)
        self.height(height)

    def scale(self, scale):
        """Scale the rectangle."""
        self.width(self.get('width') * scale)
        self.height(self.get('height') * scale)
        HasStroke.scale(self, scale)
        HasXY.scale(self, scale)
        return self

    def width(self, width):
        """Width of rectangle."""
        self.prop['width'] = width
        return self

    def height(self, height):
        """Height of rectangle."""
        self.prop['height'] = height
        return self

    # pylint: disable=invalid-name

    def rx(self, rx):
        """Rx of rectangle."""
        self.prop['rx'] = rx
        return self

    def ry(self, ry):
        """Ry of rectangle."""
        self.prop['ry'] = ry
        return self

    def chamfer(self, rect):
        """Chamfer the rectangle."""
        self.rx(rect)
        self.ry(rect)
        return self

    def fill_opacity(self, opacity):
        """Change rectangle opacity."""
        self.prop['fill-opacity'] = opacity
        return self

    def fit_text(self, text):
        """Set the text to the center of rectangle and fit text automatically."""
        text_len = len(text.text)
        text_font_size = text.font_size_num
        width = self.get('width')
        self.width(np.floor((text_len * text_font_size * 1.9 / 3) // width + 1) * width)
        text.x(self.get('x'))
        text.shift(self.get('width') / 2, 0)

    @property
    def left(self):
        """Left side of rectangle."""
        return self.get('x')

    @property
    def top(self):
        """Top side of rectangle."""
        return self.get('y')

    @property
    def right(self):
        """Right side of rectangle."""
        return self.left + self.get('width')

    @property
    def bottom(self):
        """Bottom side of rectangle."""
        return self.top + self.get('height')


class Text(HasXY, HasFill):
    """SVG text component."""

    def __init__(self, x, y, text):
        """Initialize a Text object."""
        super().__init__('text')
        self.x(x)
        self.y(y)
        self.text = text
        self.font_size_num = 14
        self.unit = 'px'
        self.tail = "</text>"
        self.font_size(self.font_size_num, self.unit)
        self.dominant_baseline('middle')
        self.text_anchor('middle')

    def scale(self, scale):
        """Scale text."""
        HasXY.scale(self, scale)
        self.font_size_num *= scale
        self.font_size(self.font_size_num, self.unit)
        return self

    def to_string(self):
        """Return the string format of text svg."""
        return ' '.join([self.head] + self._prop_to_str() + [f'>{self.text}'] + [self.tail])

    def dominant_baseline(self, dominant_baseline):
        """Set dominant baseline of text."""
        self.prop["dominant-baseline"] = dominant_baseline
        return self

    def text_anchor(self, text_anchor):
        """Change text anchor."""
        self.prop['text-anchor'] = text_anchor
        return self

    def font_size(self, font_size, unit='px'):
        """Change font size."""
        self.font_size_num = font_size
        self.prop['font-size'] = f'{font_size}{unit}'
        return self

    def font_family(self, font_family):
        """Change font family."""
        self.prop['font-family'] = font_family
        return self

    def font_weight(self, font_weight):
        """Change font weight."""
        self.prop['font-weight'] = font_weight
        return self

    @property
    def left(self):
        """Left side of text."""
        anchor = self.get('text-anchor')
        if anchor == 'middle':
            return self.get('x') - len(self.text) / 2 * self.font_size_num * 0.6
        if anchor == 'end':
            return self.get('x') - len(self.text) * self.font_size_num * 0.6
        return self.get('x')

    @property
    def right(self):
        """Right side of text."""
        anchor = self.get('text-anchor')
        if anchor == 'middle':
            return self.get('x') + len(self.text) / 2 * self.font_size_num * 0.6
        if anchor == 'start':
            return self.get('x') + len(self.text) * self.font_size_num * 0.6
        return self.get('x')

    @property
    def top(self):
        """Top side of svg."""
        baseline = self.get('dominant-baseline')
        if baseline == 'middle':
            return self.get('y') - self.font_size_num / 2
        if baseline == 'bottom':
            return self.get('y') - self.font_size_num
        return self.get('y')

    @property
    def bottom(self):
        """Bottom side of svg."""
        baseline = self.get('dominant-baseline')
        if baseline == 'middle':
            return self.get('y') + self.font_size_num / 2
        if baseline == 'top':
            return self.get('y') + self.font_size_num
        return self.get('y')


class Path(HasFill):
    """SVG path component."""

    def __init__(self, points):
        """Initialize a Path object."""
        super().__init__('path')
        self.points = np.array(points).astype(float)
        self.d()

    def d(self, points=None):  # pylint: disable=invalid-name
        """D property of path."""
        if points is None:
            points = self.points
        else:
            self.points = np.array(points).astype(float)
        d_str = f'M {points[0][0]} {points[1][0]}'
        for i in range(1, points.shape[1]):
            d_str += f' L {points[0][i]} {points[1][i]}'
        d_str += ' Z'
        self.prop['d'] = d_str
        return self

    @property
    def left(self):
        """Get left side of path."""
        return np.min(self.points[0])

    @property
    def right(self):
        """Get right side of path."""
        return np.max(self.points[0])

    @property
    def top(self):
        """Get top side of path."""
        return np.min(self.points[1])

    @property
    def bottom(self):
        """Get bottom side of path."""
        return np.max(self.points[1])

    def scale(self, scale):
        """Scale path."""
        self.points *= scale
        self.d()
        return self

    def shift(self, x, y):
        """Shift path."""
        self.points += np.array([[x], [y]])
        self.d()
        return self

    def rotate(self, angle):
        """Rotate path."""
        center_x = np.mean(self.points[0])
        center_y = np.mean(self.points[1])
        self.shift(-center_x, -center_y)
        angle = np.pi / 180 * angle
        self.points = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]) @ self.points
        self.shift(center_x, center_y)
        self.d()
        return self


class Arc(HasFill):
    """Arc component of svg."""

    def __init__(self, radius):
        """Initialize an Arc object."""
        super().__init__('path')
        self.center = np.array([radius, radius])
        self.r = radius  # pylint: disable=invalid-name
        self.d()

    def d(self):  # pylint: disable=invalid-name
        """D property of arc."""
        x, y = self.center
        radius = self.r
        d_str = f'M {x-radius} {y}'
        d_str += f' A {radius} {radius} 0 0 1 {x+radius} {y}'
        self.prop['d'] = d_str

    @property
    def left(self):
        """Get left side of arc."""
        return self.center[0] - self.r

    @property
    def right(self):
        """Get right side of arc."""
        return self.center[0] + self.r

    @property
    def top(self):
        """Get top side of arc."""
        return self.center[1] - self.r

    @property
    def bottom(self):
        """Get bottom side of arc."""
        return self.center[1]

    def scale(self, scale):
        """Scale path."""
        self.center *= scale
        self.r *= scale
        HasStroke.scale(self, scale)
        self.d()
        return self

    def shift(self, x, y):
        """Shift path."""
        self.center += np.array([x, y])
        self.d()
        return self


class Circle(HasFill):
    """Circle component of svg."""

    def __init__(self, cx, cy, radius):
        """Initialize a Circle object."""
        super().__init__('circle')
        self.cx(cx)
        self.cy(cy)
        self.r(radius)

    # pylint: disable=invalid-name

    def cx(self, cx):
        """Get cx."""
        self.prop['cx'] = cx
        return self

    def cy(self, cy):
        """Get cy."""
        self.prop['cy'] = cy
        return self

    def r(self, radius):
        """Get r."""
        self.prop['r'] = radius
        return self

    @property
    def left(self):
        """Get left side of circle."""
        return self.get('cx') - self.get('r')

    @property
    def right(self):
        """Get right side of circle."""
        return self.get('cx') + self.get('r')

    @property
    def top(self):
        """Get top side of circle."""
        return self.get('cy') - self.get('r')

    @property
    def bottom(self):
        """Get bottom side of circle."""
        return self.get('cy') + self.get('r')

    def shift(self, x, y):
        """Shift circle."""
        self.cx(self.get('cx') + x)
        self.cy(self.get('cy') + y)
        return self

    def scale(self, scale):
        """Scale circle."""
        HasStroke.scale(self, scale)
        self.r(self.get('r') * scale)
        self.cx(self.get('cx') * scale)
        self.cy(self.get('cy') * scale)
        return self


class SVGContainer:
    """Container that contain any kinds of svg component, even its self."""

    def __init__(self):
        """Initialize an SVGContainer object."""
        self.element = []

    def add(self, ele):
        """Add an svg component."""
        self.element.append(ele)

    def scale(self, scale):
        """Scale every sub element."""
        for elem in self.element:
            elem.scale(scale)
        return self

    def shift(self, x, y):
        """Shift every sub element."""
        for elem in self.element:
            elem.shift(x, y)
        return self

    def to_string(self):
        """Convert whole svg to a string."""
        return ''.join([i.to_string() for i in self.element])

    def _repr_svg_(self):
        """Magic method for rendering svg in jupyter notebook."""
        box_data = box(self)
        width = box_data['width']
        height = box_data['height']
        head = (
            f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" "
            f"xmlns:xlink=\"http://www.w3.org/1999/xlink\">"
        )
        tail = "</svg>"
        return f"{head}{self.to_string()}{tail}"

    def to_file(self, filename='circuit.svg'):
        """Save svg file."""
        string = self._repr_svg_()
        with fdopen(filename, 'w', encoding='utf-8') as fd:
            fd.writelines(string)

    @property
    def left(self):
        """Get left side of container."""
        if not self.element:
            return np.inf
        return min(e.left for e in self.element)

    @property
    def right(self):
        """Get right side of container."""
        if not self.element:
            return -np.inf
        return max(e.right for e in self.element)

    @property
    def top(self):
        """Get top side of container."""
        if not self.element:
            return np.inf
        return min(e.top for e in self.element)

    @property
    def bottom(self):
        """Get bottom side of container."""
        if not self.element:
            return -np.inf
        return max(e.bottom for e in self.element)

    def change_color(self, color):
        """Change color of every element."""
        for elem in self.element:
            elem.change_color(color)
        return self


class ObjDots(SVGContainer):
    """SVG dots on object qubits for multiply qubits gate that act on discontinue qubits."""

    def __init__(self, obj_qubits, width, svg_config):
        """Initialize an ObjDots object."""
        super().__init__()
        self.svg_config = svg_config
        self.obj_qubits = obj_qubits
        min_obj = min(obj_qubits)
        if max(obj_qubits) - min(obj_qubits) + 1 != len(obj_qubits):
            for obj in obj_qubits:
                dot = Circle(
                    0,
                    (obj - min_obj) * self.svg_config['circuit_line_distance'] + self.svg_config['gate_size'] / 2,
                    self.svg_config['obj_dot_r'],
                )
                dot.fill('black')
                self.add(dot)
                self.add(copy.deepcopy(dot).shift(width, 0))


class CtrlDots(SVGContainer):
    """SVG dot for control qubits."""

    def __init__(self, obj_qubits, ctrl_qubits, svg_config):
        """Initialize a CtrlDots object."""
        super().__init__()
        self.svg_config = svg_config
        if ctrl_qubits:
            min_obj = min(obj_qubits)
            for ctrl in ctrl_qubits:
                ctrl_dot = Circle(
                    0,
                    (ctrl - min_obj) * self.svg_config['circuit_line_distance'] + self.svg_config['gate_size'] / 2,
                    self.svg_config['ctrl_dot_r'],
                ).fill('black')
                self.add(ctrl_dot)
            min_ctrl = min(ctrl_qubits + [min_obj])
            max_ctrl = max(ctrl_qubits + [min_obj])
            line = (
                Line(0, 0, 0, (max_ctrl - min_ctrl) * self.svg_config['circuit_line_distance'])
                .stroke('black')
                .stroke_width(self.svg_config['ctrl_line_stroke_width'])
            )
            line.shift(
                0, self.svg_config['gate_size'] / 2 + (min_ctrl - min_obj) * self.svg_config['circuit_line_distance']
            )
            self.add(line)


class SVGBasicGate(SVGContainer):  # pylint: disable=too-many-instance-attributes
    """Basic style for quantum gate."""

    def __init__(self, g, svg_config):
        """Initialize an SVGBasicGate object."""
        super().__init__()
        self.svg_config = svg_config
        self.name = g.__str_in_svg__()
        self.has_dag_mask = False
        if self.name.endswith(_DAGGER_MASK):
            self.name = self.name[:-1]
            self.has_dag_mask = True
        self.obj_qubits = g.obj_qubits
        self.ctrl_qubits = g.ctrl_qubits
        self.all_qubits = self.obj_qubits + self.ctrl_qubits
        self.all_qubits_range = list(range(min(self.all_qubits), max(self.all_qubits) + 1))
        self.background_container = SVGContainer()

    def as_background(self, *svg):
        """Set the svg component as background."""
        for _ in svg:
            self.background_container.add(svg)
        return self

    def create_n_qubits_rect(self, n_qubits):
        """Create n qubits rectangle."""
        rect = Rect(
            0,
            0,
            self.svg_config['gate_size'],
            self.svg_config['gate_size'] + (n_qubits - 1) * self.svg_config['circuit_line_distance'],
        )
        rect.chamfer(self.svg_config['gate_chamfer'])
        rect.stroke(self.svg_config['npg_stroke'])
        rect.stroke_width(self.svg_config['gate_stroke_width'])
        return rect

    def create_name_text(self, name):
        """Create text component for name."""
        text = Text(0, 0, name)
        text.font_family(self.svg_config['gate_name_font_family'])
        text.font_size(self.svg_config['gate_name_font_size'])
        text.font_weight(self.svg_config['gate_name_font_weight'])
        text.fill(self.svg_config['gate_name_color'])
        return text

    def change_background(self, color):
        """Change background color."""
        self.background_container.change_color(color)
        return self

    def create_ctrl_dots(self, center, color):
        """Create control dots."""
        ctrl_dots = CtrlDots(self.obj_qubits, self.ctrl_qubits, self.svg_config).change_color(color)
        ctrl_dots.shift(center, 0)
        return ctrl_dots

    def create_obj_dots(self, width, color):
        """Create obj dots."""
        obj_dots = ObjDots(self.obj_qubits, width, self.svg_config)
        obj_dots.change_color(color)
        return obj_dots

    def move_to_create_qubit(self):
        """Move the gate to correct qubit position."""
        self.shift(0, min(self.obj_qubits) * self.svg_config['circuit_line_distance'])
        return self


def get_bound_color(svg_config, fill, stroke):
    """Get bound color."""
    if svg_config['background'] == fill:
        return stroke
    return fill


class SVGGate(SVGBasicGate):
    """SVG Gate."""

    def __init__(self, g, svg_config):
        """Initialize an SVGGate object."""
        super().__init__(g, svg_config)
        self.rect = self.create_n_qubits_rect(g.n_qubits)
        self.rect.fill(self.svg_config['npg_fill'])
        self.rect.fill_opacity(self.svg_config['npg_fill_opacity'])
        self.rect.stroke(self.svg_config['npg_stroke'])
        self.rect.stroke_width(self.svg_config['gate_stroke_width'])

        self.text = self.create_name_text(self.name)
        super_align(self.rect, self.text, 'middle', 'middle', 'h')
        self.rect.fit_text(self.text)
        self.text_container = SVGContainer()
        self.text_container.add(self.text)
        if self.has_dag_mask:
            self.dagger = self.create_name_text(_DAGGER_MASK)
            self.dagger.font_size(self.svg_config['gate_name_font_size'] * 0.7)
            self.dagger.shift(self.text.right + 6, self.text.top)
            self.text_container.add(self.dagger)
        color = get_bound_color(self.svg_config, self.svg_config['npg_fill'], self.svg_config['npg_stroke'])
        self.obj_dots = self.create_obj_dots(self.rect.right, color)
        self.ctrl_dots = self.create_ctrl_dots(self.rect.right / 2, color)
        self.add(self.ctrl_dots)
        self.add(self.rect)
        self.add(self.text_container)
        self.add(self.obj_dots)
        self.as_background(self.rect, self.ctrl_dots, self.obj_dots)
        self.move_to_create_qubit()


class SVGChannel(SVGBasicGate):
    """SVG for noise channel."""

    def __init__(self, gate, svg_config):
        """Initialize an SVGParameterGate object."""
        super().__init__(gate, svg_config)
        self.rect1 = self.create_n_qubits_rect(max(self.obj_qubits) - min(self.obj_qubits) + 1)
        self.rect1.fill(self.svg_config['noise_fill'])
        self.rect1.fill_opacity(self.svg_config['npg_fill_opacity'])
        self.rect1.stroke(self.svg_config['npg_stroke'])
        self.rect1.stroke_width(self.svg_config['gate_stroke_width'])
        self.name_text = self.create_name_text(self.name)
        self.name_text.shift(0, self.rect1.get('height') / 2 - self.svg_config['gate_size'] / 10)
        coeff_str = f"{gate.__type_specific_str__()}"

        self.coeff_text = self.create_name_text(coeff_str)
        self.coeff_text.shift(0, self.rect1.get('height') / 2 + self.svg_config['gate_size'] * 0.3)
        self.coeff_text.font_size(self.svg_config['gate_name_font_size'] * 0.7)
        self.rect1.fit_text(self.name_text)
        self.rect1.fit_text(self.coeff_text)
        self.rect1.fit_text(self.name_text)
        color = get_bound_color(self.svg_config, self.svg_config['npg_fill'], self.svg_config['npg_stroke'])
        self.obj_dots = self.create_obj_dots(self.rect1.right, color)
        self.ctrl_dots = self.create_ctrl_dots(self.rect1.right / 2, color)
        self.add(self.ctrl_dots)
        self.add(self.rect1)
        self.add(self.name_text)
        self.add(self.coeff_text)
        self.add(self.obj_dots)
        self.as_background(self.ctrl_dots, self.rect1, self.obj_dots)
        self.move_to_create_qubit()


class SVGSWAPGate(SVGBasicGate):
    """Style for swap gate."""

    def __init__(self, g, svg_config):
        """Initialize an SVGSWAPGate object."""
        super().__init__(g, svg_config)
        self.rect1 = SWAPIcon(svg_config)
        self.rect2 = SWAPIcon(svg_config)
        self.rect2.shift(0, (max(self.obj_qubits) - min(self.obj_qubits)) * self.svg_config['circuit_line_distance'])
        color = get_bound_color(self.svg_config, self.svg_config['swap_fill'], self.svg_config['swap_stroke'])
        self.swap_line = (
            Line(self.rect1.right / 2, self.rect1.right / 2, 0, self.rect2.bottom)
            .stroke_width(self.svg_config['ctrl_line_stroke_width'])
            .stroke(color)
        )

        self.ctrl_dots = self.create_ctrl_dots(self.rect1.right / 2, color)
        self.add(self.swap_line)
        self.add(self.ctrl_dots)
        self.add(self.rect1)
        self.add(self.rect2)
        self.as_background(self.rect1, self.rect2, self.swap_line, self.ctrl_dots)
        self.move_to_create_qubit()


class SVGCNOTGate(SVGBasicGate):
    """SVG for cnot gate."""

    def __init__(self, g, svg_config):
        """Initialize an SVGCNOTGate object."""
        super().__init__(g, svg_config)
        if isinstance(g, CNOTGate):
            self.obj_qubits = [g.obj_qubits[0]]
            self.ctrl_qubits = [g.obj_qubits[1]]
        self.rect1 = self.create_n_qubits_rect(1)
        self.rect1.fill(self.svg_config['cnot_fill'])
        self.rect1.fill_opacity(self.svg_config['cnot_fill_opacity'])
        self.rect1.stroke(self.svg_config['cnot_stroke'])
        self.rect1.stroke_width(self.svg_config['cnot_stroke_width'])

        self.cross = SVGContainer()
        self.cross.add(
            Line(-self.svg_config['cnot_cross_size'] / 2, self.svg_config['cnot_cross_size'] / 2, 0, 0)
            .stroke(self.svg_config['cnot_cross_stroke'])
            .stroke_width(self.svg_config['cnot_cross_stroke_width'])
        )
        self.cross.add(
            Line(0, 0, -self.svg_config['cnot_cross_size'] / 2, self.svg_config['cnot_cross_size'] / 2)
            .stroke(self.svg_config['cnot_cross_stroke'])
            .stroke_width(self.svg_config['cnot_cross_stroke_width'])
        )
        self.cross.shift(self.svg_config['gate_size'] / 2, self.svg_config['gate_size'] / 2)
        color = get_bound_color(self.svg_config, self.svg_config['cnot_fill'], self.svg_config['cnot_stroke'])
        self.ctrl_dots = self.create_ctrl_dots(self.rect1.right / 2, color)
        self.add(self.ctrl_dots)
        self.add(self.rect1)
        self.add(self.cross)
        self.as_background(self.ctrl_dots, self.rect1)
        self.move_to_create_qubit()


class SVGParameterGate(SVGBasicGate):
    """SVG for parameterized gate."""

    def __init__(self, gate, svg_config):
        """Initialize an SVGParameterGate object."""
        super().__init__(gate, svg_config)
        self.rect1 = self.create_n_qubits_rect(max(self.obj_qubits) - min(self.obj_qubits) + 1)
        self.rect1.fill(self.svg_config['pg_fill'])
        self.rect1.fill_opacity(self.svg_config['pg_fill_opacity'])
        self.rect1.stroke(self.svg_config['pg_stroke'])
        self.rect1.stroke_width(self.svg_config['gate_stroke_width'])
        self.name_text = self.create_name_text(self.name)
        self.name_text.shift(0, self.rect1.get('height') / 2 - self.svg_config['gate_size'] / 10)
        coeff_str = f"{gate.__type_specific_str__()}"

        self.coeff_text = self.create_name_text(coeff_str)
        self.coeff_text.shift(0, self.rect1.get('height') / 2 + self.svg_config['gate_size'] * 0.3)
        self.coeff_text.font_size(self.svg_config['gate_name_font_size'] * 0.7)
        self.rect1.fit_text(self.name_text)
        self.rect1.fit_text(self.coeff_text)
        self.rect1.fit_text(self.name_text)
        color = get_bound_color(self.svg_config, self.svg_config['pg_fill'], self.svg_config['pg_stroke'])
        self.obj_dots = self.create_obj_dots(self.rect1.right, color)
        self.ctrl_dots = self.create_ctrl_dots(self.rect1.right / 2, color)
        self.add(self.ctrl_dots)
        self.add(self.rect1)
        self.add(self.name_text)
        self.add(self.coeff_text)
        self.add(self.obj_dots)
        self.as_background(self.ctrl_dots, self.rect1, self.obj_dots)
        self.move_to_create_qubit()


class GateContainer(SVGContainer):
    """Container that contains gate. This container can make sure that the gate will added layer by layer."""

    def __init__(self, n_qubits, svg_config):
        """Initialize a GateContainer object."""
        super().__init__()
        self.svg_config = svg_config
        self.n_qubits = n_qubits
        self.circ_high = [0 for _ in range(n_qubits)]

    def add(self, ele):
        """Add an svg component."""
        high = max(self.circ_high[i] for i in ele.all_qubits_range)
        ele.shift(high + self.svg_config['gate_v_distance'] * (high != 0), 0)
        super().add(ele)
        width = ele.right - ele.left
        barrier_not_show = isinstance(ele, SVGBarrier) and not ele.show
        for i in ele.all_qubits_range:
            self.circ_high[i] = high + width + self.svg_config['gate_v_distance'] * (high != 0) * (not barrier_not_show)
        return self


def create_qubit_container(qubit_name, svg_config):
    """Create qubit container."""
    qubit_container = SVGContainer()
    for i, j in enumerate(qubit_name):
        qubit_container.add(SVGQubit(0, i * svg_config['circuit_line_distance'], j, svg_config))
    qubit_container.shift(svg_config['padding_x'], svg_config['padding_y'] + svg_config['gate_size'] / 2)
    return qubit_container


def create_circuit_line_container(n_qubits, circ_len, svg_config):
    """Create circuit line container."""
    circuit_line_container = SVGContainer()
    for i in range(n_qubits):
        circuit_line_container.add(
            SVGCircuitLine(
                0,
                circ_len,
                i * svg_config['circuit_line_distance'],
                i * svg_config['circuit_line_distance'],
                svg_config,
            )
        )
    return circuit_line_container


def add_to_gate_container(gate_container: GateContainer, gate, svg_config, n_qubits):
    """Add gate to gate container."""
    if isinstance(gate, BarrierGate):
        if gate.obj_qubits:
            gate_container.add(SVGBarrier(gate.obj_qubits, svg_config))
        else:
            gate_container.add(SVGBarrier(range(n_qubits), svg_config, gate.show))
    elif isinstance(gate, Measure):
        gate_container.add(SVGMeasure(gate, svg_config))
    elif isinstance(gate, XGate) and len(gate.ctrl_qubits) == 1:
        gate_container.add(SVGCNOTGate(gate, svg_config))
    elif isinstance(gate, CNOTGate):
        gate_container.add(SVGCNOTGate(gate, svg_config))
    elif isinstance(gate, SWAPGate):
        gate_container.add(SVGSWAPGate(gate, svg_config))
    elif isinstance(gate, ParameterGate) and not isinstance(gate, (TGate, SGate)):
        gate_container.add(SVGParameterGate(gate, svg_config))
    elif isinstance(gate, NoiseGate):
        gate_container.add(SVGChannel(gate, svg_config))
    else:
        gate_container.add(SVGGate(gate, svg_config))


def create_connecter(width, connect_high, arrow_size, svg_config):
    """Create connecter."""
    line_1 = (
        Line(0, width, 0, 0)
        .stroke(svg_config['circuit_line_stroke'])
        .stroke_width(svg_config['circuit_line_stroke_width'])
        .stroke_dasharray(5, 5)
    )
    line_2 = (
        Line(0, 0, -connect_high, 0)
        .stroke(svg_config['circuit_line_stroke'])
        .stroke_width(svg_config['circuit_line_stroke_width'])
        .stroke_dasharray(5, 5)
    )
    line_3 = (
        Line(0, 0, 0, connect_high)
        .stroke(svg_config['circuit_line_stroke'])
        .stroke_width(svg_config['circuit_line_stroke_width'])
        .stroke_dasharray(5, 5)
    )
    line_2.shift(width, 0)
    line_4 = (
        Line(-arrow_size, 0, -arrow_size, 0)
        .stroke(svg_config['circuit_line_stroke'])
        .stroke_width(2 * svg_config['circuit_line_stroke_width'])
    )
    line_5 = (
        Line(arrow_size, 0, -arrow_size, 0)
        .stroke(svg_config['circuit_line_stroke'])
        .stroke_width(2 * svg_config['circuit_line_stroke_width'])
    )
    line_4.shift(0, connect_high)
    line_5.shift(0, connect_high)
    out = SVGContainer()
    out.add(line_1)
    out.add(line_2)
    out.add(line_3)
    out.add(line_4)
    out.add(line_5)
    out.shift(-out.left, -out.top)
    return out


class SVGCircuit(SVGContainer):
    """SVG circuit component."""

    # pylint: disable=too-many-locals
    def __init__(self, circuit, svg_config, len_limit):
        """Initialize an SVGCircuit object."""
        super().__init__()
        self.svg_config = svg_config
        _check_input_type("circuit", Circuit, circuit)
        old_qubits = sorted(circuit.all_qubits.keys())
        qubit_container = create_qubit_container(old_qubits, self.svg_config)
        qubit_container_right = qubit_container.right

        circuit = circuit.compress()
        n_qubits = circuit.n_qubits
        gate_containers = [GateContainer(n_qubits, svg_config)]
        try_gate_container = GateContainer(n_qubits, svg_config)
        circ_split_dist = svg_config['circuit_line_distance'] * 0.2
        for gate in circuit:
            add_to_gate_container(try_gate_container, gate, self.svg_config, n_qubits)
            try_width = try_gate_container.right - try_gate_container.left + 2 * self.svg_config['gate_start_distance']
            if try_width + qubit_container_right >= len_limit:
                gate_containers.append(GateContainer(n_qubits, svg_config))
                try_gate_container = GateContainer(n_qubits, svg_config)
            add_to_gate_container(gate_containers[-1], gate, self.svg_config, n_qubits)
        max_len = max(
            gate_container.right - gate_container.left + 2 * self.svg_config['gate_start_distance']
            for gate_container in gate_containers
        )
        connecter = create_connecter(max_len, 24, 10, self.svg_config)
        connecter.shift(qubit_container.right, 0)
        circuit_line_container = create_circuit_line_container(n_qubits, max_len, self.svg_config)
        circuit_line_container.shift(
            qubit_container.right, self.svg_config['padding_y'] + self.svg_config['gate_size'] / 2
        )
        for gate_container in gate_containers:
            gate_container.shift(
                circuit_line_container.left + self.svg_config['gate_start_distance'], self.svg_config['padding_y']
            )
        self.front = SVGContainer()
        bottom = 0
        for i, gate_container in enumerate(gate_containers):
            q_container = copy.deepcopy(qubit_container)
            if i != 0:
                conn = copy.deepcopy(connecter)
                conn.shift(0, i * circ_split_dist + bottom)
            if i != 0:
                self.front.add(conn)
                bottom = self.front.bottom
            q_container.shift(0, (i != 0) * circ_split_dist + bottom)
            cl_container = copy.deepcopy(circuit_line_container)
            cl_container.shift(0, (i != 0) * circ_split_dist + bottom)
            gate_container.shift(0, (i != 0) * circ_split_dist + bottom)
            self.front.add(q_container)
            self.front.add(cl_container)
            self.front.add(gate_container)
            bottom = self.front.bottom
        front_box = box(self.front)
        self.background = Rect(
            front_box['left'],
            front_box['top'],
            front_box['width'] + self.svg_config['gate_size'],
            front_box['height'] + self.svg_config['gate_size'],
        )
        self.background.fill(self.svg_config['background'])
        self.front.shift(self.svg_config['gate_size'] / 2, self.svg_config['gate_size'] / 2)
        self.add(self.background)
        self.add(self.front)


class SVGQubit(Text):
    """SVT qubit component."""

    def __init__(self, x, y, idx, svg_config):
        """Initialize an SVTQubit object."""
        super().__init__(x, y, f"q{idx}:")
        self.svg_config = svg_config
        self.text_anchor('start')
        self.font_family(self.svg_config['gate_name_font_family'])
        self.font_size(self.svg_config['qubit_font_size'])
        self.font_weight(self.svg_config['qubit_name_font_weight'])
        self.fill(self.svg_config['qubit_name_color'])


class SVGCircuitLine(Line):
    """SVG circuit line."""

    def __init__(self, x1, x2, y1, y2, svg_config):  # pylint: disable=too-many-arguments
        """Initialize an SVGCircuitLine object."""
        super().__init__(x1, x2, y1, y2)
        self.svg_config = svg_config
        self.stroke(self.svg_config['circuit_line_stroke'])
        self.stroke_width(self.svg_config['circuit_line_stroke_width'])


class SVGBarrier(SVGContainer):  # pylint: disable=too-many-instance-attributes
    """SVG barrier."""

    def __init__(self, obj_qubits, svg_config, show=True):
        """Initialize an SVGBarrier object."""
        super().__init__()
        self.svg_config = svg_config
        self.n_qubits = len(obj_qubits)
        self.obj_qubits = obj_qubits
        self.ctrl_qubits = []
        self.all_qubits = self.obj_qubits
        self.all_qubits_range = self.obj_qubits
        self.show = show
        self.rect = Rect(
            0,
            0,
            self.svg_config['barrier_width'] * show,
            self.svg_config['gate_size'] + (self.n_qubits - 1) * self.svg_config['circuit_line_distance'],
        )
        self.rect.fill(self.svg_config['barrier_fill'])
        self.rect.fill_opacity(self.svg_config['barrier_opacity'])
        self.add(self.rect)
        self.shift(0, min(self.obj_qubits) * self.svg_config['circuit_line_distance'])


class SVGMeasure(SVGBasicGate):
    """SVG for measure gate."""

    def __init__(self, g, svg_config):
        """Initialize an SVGMeasure object."""
        super().__init__(g, svg_config)
        self.rect = self.create_n_qubits_rect(1)
        self.rect.fill(self.svg_config['measure_fill'])
        self.rect.fill_opacity(self.svg_config['measure_fill_opacity'])
        self.rect.stroke(self.svg_config['measure_stroke'])
        self.rect.stroke_width(self.svg_config['gate_stroke_width'])
        self.circle = Circle(0.5 * self.svg_config['gate_size'], 0.8 * self.svg_config['gate_size'], 2).fill(
            self.svg_config['measure_icon_color']
        )
        self.arc = Arc(self.svg_config['gate_size'] * 0.4)
        self.arc.stroke(self.svg_config['measure_icon_color'])
        self.arc.stroke_width(self.svg_config['measure_arc_stroke_width'])
        self.arc.fill_opacity(0)
        super_align(self.rect, self.arc, 0.8, 'bottom', relative=True)
        super_align(self.rect, self.arc, 'middle', 'middle', 'v')
        self.arrow = Arrow(0.2).fill(self.svg_config['measure_icon_color'])
        self.arrow.scale(6).rotate(30)
        super_align(
            self.circle, self.arrow, 'middle', self.arrow.left + 3 * self.arrow.w * np.cos(np.pi / 180 * 30), 'v'
        )
        super_align(self.rect, self.arrow, 0.8, 'bottom', relative=True)
        self.icon = SVGContainer()
        self.icon.add(self.circle)
        self.icon.add(self.arc)
        self.icon.add(self.arrow)
        self.icon.scale(self.svg_config['measure_icon_scale'])
        super_align(self.rect, self.icon, 'middle', 'middle', 'v')
        super_align(self.rect, self.icon, 0.8, 'bottom', relative=True)
        self.add(self.rect)
        self.add(self.icon)
        if g.reset_to is not None:
            text = Text(0, 0, f"{g.reset_to}")
            text.font_family(self.svg_config['gate_name_font_family'])
            text.font_size(self.svg_config['gate_name_font_size'] * 0.7)
            text.font_weight(self.svg_config['gate_name_font_weight'])
            text.fill(self.svg_config['gate_name_color'])
            self.text = text
            self.add(self.text)
            super_align(self.rect, self.text, 0.05, 'left', relative=True)
            super_align(self.rect, self.text, 0.05, 'top', relative=True)
        self.as_background(self.rect)
        self.move_to_create_qubit()


class Arrow(Path):
    """Arrow component."""

    def __init__(self, width=0.3):
        """Initialize an Arrow object."""
        self.w = width  # pylint: disable=invalid-name
        self.points = [
            [0, 1, 2, 1 + width, 1 + width, 1 - width, 1 - width],
            [np.sqrt(3), 0, np.sqrt(3), np.sqrt(3), 5, 5, np.sqrt(3)],
        ]
        super().__init__(self.points)


class SWAPIcon(SVGContainer):
    """Icon for swap gate."""

    def __init__(self, svg_config):
        """Initialize a SWAPIcon object."""
        super().__init__()
        self.svg_config = svg_config
        self.rect = Rect(0, 0, self.svg_config['gate_size'], self.svg_config['gate_size'])
        self.rect.chamfer(self.svg_config['gate_chamfer'])
        self.rect.fill(self.svg_config['swap_fill'])
        self.rect.fill_opacity(self.svg_config['swap_fill_opacity'])
        self.rect.stroke(self.svg_config['swap_stroke'])
        self.rect.stroke_width(self.svg_config['gate_stroke_width'])

        self.path = Arrow().scale(40)
        self.path2 = copy.deepcopy(self.path).rotate(180)
        super_align(self.path, self.path2, 'right', 'left')
        super_align(self.path, self.path2, 'top', 'top')
        self.icon = SVGContainer()
        self.icon.add(self.path)
        self.icon.add(self.path2)
        self.icon.scale(self.svg_config['gate_size'] / 200).scale(self.svg_config['swap_icon_ratio'])
        self.icon.change_color(self.svg_config['swap_icon_color'])
        center_align_to(self.rect, self.icon)
        self.add(self.rect)
        self.add(self.icon)


def box(svg):
    """Get the left, right, top, bottom, width, height, cx, and cy of the given svg element."""
    res = {}
    res['left'] = svg.left
    res['right'] = svg.right
    res['top'] = svg.top
    res['bottom'] = svg.bottom
    res['width'] = res['right'] - res['left']
    res['height'] = res['bottom'] - res['top']
    res['cx'] = res['left'] + res['width'] / 2
    res['cy'] = res['top'] + res['height'] / 2
    return res


def center_align_to(target, source):
    """Align source svg to center of target svg."""
    t_x = (target.right - target.left) / 2
    t_y = (target.bottom - target.top) / 2
    s_x = (source.right - source.left) / 2
    s_y = (source.bottom - source.top) / 2
    source.shift(t_x - s_x, t_y - s_y)


def super_align(target, source, t_des, s_des, direction=None, relative=False):  # pylint: disable=too-many-arguments
    """
    Align.

    Args:
        target: ???
        source: ???
        t_des ([str, numbers.Number]): if str, should be 'left', 'right', 'top', 'bottom' or 'middle'.
        s_des ([str, numbers.Number]): if str, should be 'left', 'right', 'top', 'bottom' or 'middle'.
        direction (str): should 'v' for vertical or 'h' for horizontal.
        relative (bool): if true, t_des and s_des should be number and will be regard as a percentage.
    """

    def relative_to_num(des, direction, box_data):
        """Relative keyword to absolute number."""
        if direction == 'h':
            return des * box_data['height']
        return des * box_data['width']

    def str_to_num(des, direction, box_data):
        """Convert string keyword to absolute number."""
        if des in ['left', 'right']:
            if direction == 'h':
                raise ValueError('wrong direction')
            return box_data[des]
        if des in ['top', 'bottom']:
            if direction == 'v':
                raise ValueError('wrong direction')
            return box_data[des]
        if direction == 'h':
            return box_data['cy']
        return box_data['cx']

    def determine_direction(t_des, s_des, direction):
        """Check the direction."""
        v_key = ['left', 'right']
        h_key = ['top', 'bottom']
        if t_des in v_key and s_des in h_key or t_des in h_key and s_des in v_key:
            raise ValueError("Wrong direction")
        if t_des in v_key or s_des in v_key:
            return 'v'
        if t_des in h_key or s_des in h_key:
            return 'h'
        return direction

    tar_box = box(target)
    sour_box = box(source)
    direction = determine_direction(t_des, s_des, direction)
    if isinstance(t_des, str):
        t_des = str_to_num(t_des, direction, tar_box)
    elif relative:
        t_des = relative_to_num(t_des, direction, tar_box)

    if isinstance(s_des, str):
        s_des = str_to_num(s_des, direction, sour_box)
    elif relative:
        s_des = relative_to_num(s_des, direction, sour_box)
    vec = t_des - s_des
    if direction == 'v':
        source.shift(vec, 0)
    else:
        source.shift(0, vec)
