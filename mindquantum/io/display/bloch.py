# -*- coding: utf-8 -*-
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
"""Bloch module."""

__all__ = ['SQBloch']

import numpy as np
import matplotlib.pyplot as plt
from mindquantum.utils.type_value_check import _num_type
from ._config import _bloch_drawer_config

_bloch_config_type = {
    'radius': _num_type,
    'ca_width': _num_type,
    'ca_color': str,
    'sp_alpha': _num_type,
    'sp_color': str,
    'sp_width': _num_type,
    'arr_color': str,
    'arr_width': _num_type,
    'arr_size': _num_type,
    'label': [3, str, 1],
    'label_size': _num_type,
    'label_color': str,
    'ax_offset': [2, _num_type, 0],
    "arr_alpha": bool,
}

class SQBloch:
    """
    Bloch sphere for Single qubit.

    Args:
        radius (float): Radius of bloch. Default: 1.
        ca_width (float): The width of coordinate axis. Default: 1.
        ca_color (str): The color of coordinate axis. Default: "gray".
        sp_alpha (float): Sphere transparency. Default: 0.2.
        sp_color (str): The color of sphere. Default: "silver".
        sp_width (float): The width of spherical curve. Default: 1.
        arr_color (str): The color of arrow. Default: "red".
        arr_width (float): The width of arrow. Default: 3.
        arr_size (float): The size of arrow(arrow length ratio). Default: 0.1.
        label (list): The label of coordinate axis. Default: ["x", "y", "z"].
        label_size (int): Label font size. Default: 20.
        label_color (str): Label font color. Default: "black".
        ax_offset (tuple or list): Axis offset(rotating coordinate system). Default: (0, 0).
        arr_alpha (bool): Whether the arrow transparency change. Default: True.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> from mindquantum.io.display.bloch import SQBloch
        >>> ax = plt.figure().add_subplot(projection='3d')
        >>> b = SQBloch() # Generate a bloch with default configuration.
        >>> b.plot(ax, [1, 0, 0])
        >>> plt.show()
        >>> b = SQBloch({'label':["x", "y", ['$\\left|0\\right>$', '$\\left|1\\right>$']],
                         'label_size':22,
                         'label_color':'blue'},
                         arr={'arr_color':'g', 'arr_width':2, 'arr_size':0.2},
                         radius=2,) # Set configuration.
        >>> b.revise_config({'sp_alpha':0.25, 'sp_width':2.5, 'sp_color':'Yellow'}) # Revise configuration.
        >>> b.plot(ax, [1, 0, 0])
        >>> plt.show()
        >>> b.plot(ax, [0, 0, 0]) # Bloch only.
        >>> plt.show()
    """
    def __init__(self, *args, **kwargs):
        self.ax = None
        self.para = {}
        self.revise_config(*args, **kwargs)

    def fill_config(self):
        """Fill configuration."""
        self.para["radius"] = self.para.get("radius", 1)                 # bloch radius
        self.para["ca_width"] = self.para.get("ca_width", 1)             # the width of coordinate axis
        self.para["ca_color"] = self.para.get("ca_color", "gray")        # the color of coordinate axis
        self.para["sp_alpha"] = self.para.get("sp_alpha", 0.2)           # sphere transparency
        self.para["sp_color"] = self.para.get("sp_color", "silver")      # the color of sphere
        self.para["sp_width"] = self.para.get("sp_width", 1)             # the width of spherical curve
        self.para["arr_color"] = self.para.get("arr_color", "red")       # the color of arrow
        self.para["arr_width"] = self.para.get("arr_width", 3)           # the width of arrow
        self.para["arr_size"] = self.para.get("arr_size", 0.1)           # the size of arrow(arrow length ratio)
        self.para["label"] = self.para.get("label", ["x", "y", "z"])     # the label of coordinate axis
        self.para["label_size"] = self.para.get("label_size", 20)        # label font size
        self.para["label_color"] = self.para.get("label_color", "black") # label font color
        self.para["ax_offset"] = self.para.get("ax_offset", (0, 0))      # axis offset
        self.para["arr_alpha"] = self.para.get("arr_alpha", True)        # change in arrow transparency

    def check_config(self):
        """Check configuration type."""
        for k, v in _bloch_config_type.items():
            self._check_config_type(k, v, default=_bloch_drawer_config.get(k, None))

    def _check_config_type(self, k, require_type, default):
        """Check config type."""
        v = self.para.get(k, None)
        if isinstance(require_type, list):
            try:
                if not isinstance(v, (list, tuple)):
                    raise ValueError
                if len(v) != require_type[0]:
                    raise ValueError
                for i in range(require_type[0]):
                    self.__check_config_type_listitems(v[i], require_type)
            except ValueError:
                self._config_warning(k, v)
                self.para[k] = default
            return
        if not isinstance(v, require_type):
            self._config_warning(k, v)
            self.para[k] = default
    def __check_config_type_listitems(self, v, require_type):
        if require_type[2] and isinstance(v, (list, tuple)):
            for i in v:
                if not isinstance(i, require_type[1]):
                    raise ValueError
        elif not isinstance(v, require_type[1]):
            raise ValueError

    def _config_warning(self, k, v):
        """Config warning."""
        print("Warning: '{}' got an unexpected argument '{}'!".format(k, v))

    def __str__(self):
        """Config information of bloch."""
        itms = ["radius", "ca_width", "ca_color", "ax_offset",
                "sp_alpha", "sp_color", "sp_width",
                "arr_color", "arr_width", "arr_size", "arr_alpha",
                "label", "label_size", "label_color",]
        config = "< Bloch Configuration ::\r\n"
        for i in itms:
            config += " * " + i + " : " + str(self.para.get(i, "None")) + "\r\n"
        config += "::>"
        return config

    def _prase_dict(self, dc):
        """Prase configuration."""
        for k, v in dc.items():
            if isinstance(v, dict):
                self._prase_dict(v)
            else:
                if not isinstance(k, str):
                    raise TypeError("Parameter name should be a string, but get {}!".format(type(k)))
                self.para[k] = v

    def revise_config(self, *args, **kwargs):
        """Revise configuration."""
        self._prase_dict(kwargs)
        for p in args:
            if isinstance(p, dict):
                self._prase_dict(p)
        self.fill_config()
        self.check_config()

    def show(self, ax):
        """
        Plot bloch only.
        """
        self.ax = ax
        self.ax.grid(False) # 隐藏网格线
        self.ax.set_axis_off() # 隐藏坐标轴
        self.draw_axis() # 绘制坐标轴
        self.draw_sphere() # 绘制球体
        self.draw_label() # 添加标签

    def plot(self, ax, vectors):
        """
        Plot bloch with vectors.

        Args:
        ax (matplotlib.axes._subplots.Axes3DSubplot): 3D canvas.
        vectors (list or tuple): Three dimensional coordinates.

        Raises:
            ValueError: If the size of `vectors` is inappropriate.

        Examples:
            >>> import matplotlib.pyplot as plt
            >>> from mindquantum.io.display.bloch import SQBloch
            >>> ax = plt.figure().add_subplot(projection='3d')
            >>> b = SQBloch()
            >>> vec = [0, 0, 1]
            >>> b.plot(ax, vec)
            >>> plt.show()
            >>> vecs = [[0, 0, 1], [0, 0, 1], [1, 0, 0]]
            >>> b.plot(ax, vecs)
            >>> plt.show()
        """
        vectors = self._check_vectors_type(vectors)
        self.show(ax)
        if len(vectors.shape) == 1:
            self.draw_vector(vectors, 1) # 绘制向量
        else:
            alpha = 1
            d_ = 0
            if self.para.get("arr_alpha", False):
                d_ = 0.9 / len(vectors)
            for v in vectors[::-1]:
                self.draw_vector(v, alpha)
                alpha -= d_

    def _check_vectors_type(self, v):
        """Check vectors type"""
        if not isinstance(v, (list, tuple, np.ndarray)):
            raise TypeError("Type must be a list, tuple or np.ndarray, but get {}".format(type(v)))
        v = np.array(v).squeeze()
        s = v.shape
        try:
            if s[-1] != 3 or len(s)>2:
                raise ValueError("Invalid vectors!")
        except (TypeError, IndexError):
            raise ValueError("Invalid vectors!")
        return v

    def draw_axis(self):
        """Draw the axis of bloch."""
        r = self.para.get("radius", 1) # bloch radius
        ca = np.array([-r, r]) # coordinate axis
        coc = np.array([0, 0]) # the centre of a circle
        lw = self.para.get("ca_width", "gray")
        color = self.para.get("ca_color", "gray")

        self.ax.plot(ca, coc, zdir="z", label="X", lw=lw, color=color)
        self.ax.plot(coc, ca, zdir="z", label="Y", lw=lw, color=color)
        self.ax.plot(coc, ca, zdir="x", label="Z", lw=lw, color=color)

        r *= 1.3
        self.ax.set_xlim3d(-r, r)
        self.ax.set_ylim3d(-r, r)
        self.ax.set_zlim3d(-r, r)

    def draw_sphere(self):
        """Draw the sphere of bloch."""
        r = self.para.get("radius", 1)
        u = np.linspace(0, np.pi, 80)
        v = np.linspace(0, np.pi, 80)
        x = r * np.outer(np.cos(u), np.sin(v))
        y = r * np.outer(np.sin(u), np.sin(v))
        z = r * np.outer(np.ones(np.size(u)), np.cos(v))

        color1 = self.para.get("sp_color", "silver")
        color2 = self.para.get("ca_color", "gray")
        alpha = self.para.get("sp_alpha", 0.2)
        lw = self.para.get("sp_width", 1)

        self.ax.plot_surface(x, y, z, color=color1, alpha=alpha)
        self.ax.plot_surface(-x, -y, z, color=color1, alpha=alpha)
        self.ax.plot_wireframe(x, y, z, rstride=16, cstride=16, color=color2, alpha=alpha)
        self.ax.plot_wireframe(-x, -y, z, rstride=16, cstride=16, color=color2, alpha=alpha)
        #
        c = r * np.cos(u)
        s = r * np.sin(u)
        c = np.concatenate((-c, c))
        s = np.concatenate((-s, s))

        self.ax.plot(c, s, zdir="x", lw=lw, color=color2)
        self.ax.plot(c, s, zdir="y", lw=lw, color=color2)
        self.ax.plot(c, s, zdir="z", lw=lw, color=color2)

    def draw_label(self):
        """Draw the label of bloch."""
        r = self.para.get("radius", 1) * 1.2
        fsize = self.para.get("label_size", 20)
        color = self.para.get("label_color", "black")
        label = self.para.get("label", ['x', 'y', 'z'])

        self._draw_label(label[0], r, 0, 0, fsize, color, 'x')
        self._draw_label(label[1], 0, r, 0, fsize, color, 'y')
        self._draw_label(label[2], 0, 0, r, fsize, color, 'z')

        ao = self.para.get("ax_offset", (0, 0))
        self.ax.view_init(30 + ao[0], 45 + ao[1])

    def _draw_label(self, l, x, y, z, s, c, noe):
        """Draw the label of bloch."""
        if isinstance(l, str):
            self.ax.text(x, y, z, l, size=s, color=c, ha="center", va="center")
            return
        try:
            self.ax.text(-x, -y, -z, l[1], size=s, color=c, ha="center", va="center")
        finally:
            try:
                self.ax.text(x, y, z, l[0], size=s, color=c, ha="center", va="center")
            except (TypeError, IndexError):
                self.ax.text(x, y, z, noe, size=s, color=c, ha="center", va="center")

    def draw_vector(self, v, a):
        """Draw the vector on bloch."""
        r = self.para.get("radius", 1)
        color = self.para.get("arr_color", "red")
        lw = self.para.get("arr_width", 3)
        alr = self.para.get("arr_size", 0.1)

        self.ax.quiver(0, 0, 0, v[0], v[1], v[2], length=r, normalize=True,
                       color=color, lw=lw, arrow_length_ratio=alr, alpha=a)

def _main():
    """Test."""
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    vec = [1, 0, 0]
    b = SQBloch()
    b.plot(ax, vec)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    vec = [[1, 0, 0],
           [0, 1, 0],
           [0, 0, 1]]
    b = SQBloch()
    b.plot(ax2, vec)
    plt.show()

if __name__ == '__main__':
    _main()
