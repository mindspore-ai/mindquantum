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
"""Display one qubit state in bloch sphere."""

import numbers
import warnings
from collections import deque

import numpy as np
import rich
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform

from mindquantum.io.display._config import (
    _bloch_default_style_,
    _bloch_default_style_dark_,
)
from mindquantum.utils.type_value_check import _check_input_type, _check_int_type


class Arrow3D(FancyArrowPatch):
    """3D arrow."""

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):  # pylint: disable=too-many-arguments
        """Initialize an Arrow3D object."""
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        """Draw artist."""
        # pylint: disable=invalid-name
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, _ = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):  # pylint: disable=unused-argument
        """Do 3d projection."""
        # pylint: disable=invalid-name
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


class BlochScene:  # pylint: disable=too-many-instance-attributes
    """
    Display a one qubit quantum state in bloch sphere.

    Args:
        config (Union[dict, str]): The bloch sphere style configuration. If ``None``, a built-in style configuration
            will be used. Beside built-in style, we also support a ``'dark'`` style. Default: ``None``.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from mindquantum.core.gates import RX, RZ
        >>> from mindquantum.io.display import BlochScene
        >>> state = np.array([1, 1 + 1j])/np.sqrt(3)
        >>> scene = BlochScene()
        >>> fig, ax = scene.create_scene()
        >>> scene.add_state(ax, state)
        >>> plt.show()
        >>> n_step = 100
        >>> amps = np.zeros((n_step, 2), dtype=np.complex128)
        >>> for i, angle in enumerate(np.linspace(0, np.pi * 2, n_step)):
        ...     state = RZ(angle).matrix() @ RX(np.pi / 4).matrix() @ np.array([[1], [0]])
        ...     state = state.T[0]
        ...     amps[i] = state
        >>> scene = BlochScene('dark')
        >>> fig, ax = scene.create_scene()
        >>> scene.add_state(ax, np.array([1, 1 - 1j])/np.sqrt(3), with_proj=False)
        >>> objs = scene.add_state(ax, amps[0], linecolor='r')
        >>> anim = scene.animation(fig, ax, objs, amps,history_len=10)
        >>> plt.show()
    """

    def __init__(self, config=None):
        """Initialize a BlochScene object."""
        supported_style = {'default': _bloch_default_style_, 'dark': _bloch_default_style_dark_}
        if config is None:
            config = 'default'
        if isinstance(config, str):
            if config not in supported_style:
                raise ValueError(f"Support style: {list(supported_style.keys())}, but get {config}")
            config = supported_style.get(config)
        _check_input_type("config", (dict, str), config)
        self.config = config
        self.c_ang = np.linspace(0, 2 * np.pi, 100)
        self.c_x = np.cos(self.c_ang)
        self.c_y = np.sin(self.c_ang)
        self.c_z = np.zeros_like(self.c_x)
        self.plane_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        self.plane_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        self.plane_z = np.zeros((3, 3))

    def add_ket_label(self, ax, *args, fontsize=None, **kwargs):
        """
        Set ket label in given axes.

        Args:
            ax (mpl_toolkits.mplot3d.axes3d.Axes3D): The three dimension axes you
                want add ket label.
            args (tuple): The args for ket label of text object in matplotlib.
            kwargs (dict): The key word args for ket label of text object in matplotlib.
            fontsize (int): The fontsize of label. If ``None``, the fontsize will be
                found in config of `BlochScene` with key `ket_label_fs`. Default: ``None``.
        """
        if fontsize is None:
            fontsize = self.config['ket_label_fs']
        _check_input_type("ax", Axes3D, ax)
        _check_int_type("fontsize", fontsize)
        ax.text(0, 0, 1.1, r'$\hat{{z}}=\left|0\right>$', *args, fontsize=fontsize, **kwargs)
        ax.text(0, 1.2, -0.1, r'$\hat{{y}}$', *args, fontsize=fontsize, **kwargs)
        ax.text(1.0, 0, -0.2, r'$\hat{{x}}$', *args, fontsize=fontsize, **kwargs)

    def circle_yz(self, ax, *args, angle=0, **kwargs):
        """
        Plot circle in yz plane.

        Args:
            ax (mpl_toolkits.mplot3d.axes3d.Axes3D): The three dimension axes you
                want add ket label.
            args (tuple): The args of `Axes3D.plot`.
            angle (numbers.Number): Rotate angle of circle around z axis. Default: ``0``.
            kwargs (dict): The key word args for `Axes3D.plot`.
        """
        _check_input_type("ax", Axes3D, ax)
        _check_input_type("angle", numbers.Number, angle)
        ax.plot(self.c_x * np.sin(angle), self.c_x * np.cos(angle), self.c_y, *args, **kwargs)

    def circle_xy(self, ax, *args, angle=np.pi / 2, **kwargs):
        """
        Plot circle in xy plane.

        Args:
            ax (mpl_toolkits.mplot3d.axes3d.Axes3D): The three dimension axes you
                want add circle parallel with xy plane.
            args (tuple): The args of `Axes3D.plot`.
            angle (numbers.Number): Elevation angle of circle along z axis. Default: ``π / 2``.
            kwargs (dict): The key word args for `Axes3D.plot`.
        """
        _check_input_type("ax", Axes3D, ax)
        _check_input_type("angle", numbers.Number, angle)
        ax.plot(self.c_x * np.sin(angle), self.c_y * np.sin(angle), np.cos(angle), *args, **kwargs)

    # pylint: disable=too-many-arguments
    def plot_slice(
        self,
        ax,
        x,
        y,
        z,
        frame_color,
        frame_alpha,
        surface_color,
        surface_alpha,
        frame_args=None,
        frame_kwargs=None,
        surface_args=None,
        surface_kwargs=None,
    ):
        """
        Plot reference surface in xy, yz and zx plane.

        Args:
            ax (mpl_toolkits.mplot3d.axes3d.Axes3D): The three dimension axes you
                want to add reference surface.
            x (numpy.ndarray): The x coordinate of reference plane.
            y (numpy.ndarray): The y coordinate of reference plane.
            z (numpy.ndarray): The z coordinate of reference plane.
            frame_color (str): The wire frame color.
            frame_alpha (numbers.Number): The frame transparency.
            surface_color (str): The surface color.
            surface_alpha (numbers.Number): The surface transparency.
            frame_args (tuple): The other args for `Axes3D.plot_wireframe`.
            frame_kwargs (dict): The other key word args for `Axes3D.plot_wireframe`.
            surface_args (tuple): The other args for `Axes3D.plot_surface`.
            surface_kwargs (dict): The other key word args for `Axes3D.plot_surface`.
        """
        _check_input_type("ax", Axes3D, ax)
        if frame_args is None:
            frame_args = ()
        if frame_kwargs is None:
            frame_kwargs = {}
        if surface_args is None:
            surface_args = ()
        if surface_kwargs is None:
            surface_kwargs = {}
        _check_input_type("frame_args", tuple, frame_args)
        _check_input_type("surface_args", tuple, surface_args)
        _check_input_type("frame_kwargs", dict, frame_kwargs)
        _check_input_type("surface_kwargs", dict, surface_kwargs)
        ax.plot_wireframe(x, y, z, *frame_args, color=frame_color, alpha=frame_alpha, **frame_kwargs)
        ax.plot_surface(x, y, z, *surface_args, color=surface_color, alpha=surface_alpha, **surface_kwargs)

    def set_view(self, ax, elev=0, azim=0):
        """
        Fit the view to bloch sphere.

        Args:
            ax (mpl_toolkits.mplot3d.axes3d.Axes3D): The three dimension axes you
                want to set view.
            elev (numbers.Number): stores the elevation angle in the z plane (in degrees).
                Default: ``0``.
            azim (numbers.Number): stores the azimuth angle in the (x, y) plane (in degrees).
                Default: ``0``.
        """
        ax.set_xlim3d(-1, 1)
        ax.set_ylim3d(-1, 1)
        ax.set_zlim3d(-1, 1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=elev, azim=azim)

    def gen_fig_ax(self, boxes=None):
        """
        Add three dimension scene.

        Args:
            boxes (list): A float list with 4 elements that are left, bottom, width, height
                of this scene. If ``None``, then left and bottom will be 0 and width and height
                will be 1. Default: ``None``.
        """
        if boxes is None:
            boxes = [0, 0, 1, 1]
        _check_input_type("boxes", list, boxes)
        fig = plt.figure()
        ax = fig.add_axes(boxes, projection='3d')
        return fig, ax

    def add_3d_arrow(self, ax, data, *args, **kwargs):
        """
        Add a three dimension arrow in given axes.

        Args:
            ax (mpl_toolkits.mplot3d.axes3d.Axes3D): The three dimension axes you
                want set 3d arrow.
            data (list): A 6 elements list that include start point coordinate (first three)
                and the displacement of this arrow.
            args (tuple): The other args for FancyArrowPatch.
            kwargs (dict): The other key word args for FancyArrowPatch.
        """
        arrow = Arrow3D(*data, *args, **kwargs)
        ax.add_artist(arrow)
        return arrow

    def create_scene(self):
        """Create default layout with `BlochScene.config`."""
        fig, ax = self.gen_fig_ax()
        arrowstyle = self.config['arrowstyle']
        mutation_scale = self.config['mutation_scale']
        linestyle = self.config['arrow_ls']
        delta = self.config['axis_delta']
        self.add_3d_arrow(
            ax,
            [0, 0, -1 - delta, 0, 0, 2 + 2 * delta],
            arrowstyle=arrowstyle,
            mutation_scale=mutation_scale,
            linestyle=linestyle,
        ).set_color(self.config['arrow_c'])
        self.add_3d_arrow(
            ax,
            [0, -1 - delta, 0, 0, 2 + 2 * delta, 0],
            arrowstyle=arrowstyle,
            mutation_scale=mutation_scale,
            linestyle=linestyle,
        ).set_color(self.config['arrow_c'])
        self.add_3d_arrow(
            ax,
            [-1 - delta, 0, 0, 2 + 2 * delta, 0, 0],
            arrowstyle=arrowstyle,
            mutation_scale=mutation_scale,
            linestyle=linestyle,
        ).set_color(self.config['arrow_c'])
        plane_alpha = self.config['plane_alpha']
        xy_plane_color = self.config['xy_plane_color']
        yz_plane_color = self.config['yz_plane_color']
        zx_plane_color = self.config['zx_plane_color']
        frame_alpha = self.config['frame_alpha']
        self.plot_slice(
            ax, self.plane_x, self.plane_y, self.plane_z, xy_plane_color, frame_alpha, xy_plane_color, plane_alpha
        )
        self.plot_slice(
            ax, self.plane_x, self.plane_z, self.plane_y, yz_plane_color, frame_alpha, yz_plane_color, plane_alpha
        )
        self.plot_slice(
            ax, self.plane_z, self.plane_x, self.plane_y, zx_plane_color, frame_alpha, zx_plane_color, plane_alpha
        )

        for angle in np.linspace(0, np.pi, 7):
            self.circle_xy(ax, '--', angle=angle, color='#cdcdcd', linewidth=1)
        for angle in np.linspace(0, 2 * np.pi, 4):
            self.circle_yz(ax, '--', angle=angle, color='#cdcdcd', linewidth=1)
        self.circle_xy(ax, c='#999999')
        self.circle_yz(ax, c='#999999')
        self.circle_yz(ax, c='#999999', angle=np.pi / 2)
        self.set_view(ax, elev=10, azim=40)
        self.add_ket_label(ax, c=self.config['axis_label_c'])
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.set_axis_off()
        fig.set_facecolor(self.config['fig_color'])
        ax.set_facecolor(self.config['fig_color'])
        fig.set_size_inches(self.config['fig_w'], self.config['fig_h'])
        return fig, ax

    @staticmethod
    def state_to_cor(amp: np.ndarray):
        """
        Convert one qubit state to three dimension coordinate.

        Args:
            amp (numpy.ndarray): One qubit quantum state.

        Returns:
            numpy.ndarray, three dimension coordinate.
        """
        _check_input_type('amp', np.ndarray, amp)
        if amp.shape != (2,):
            raise ValueError(f"amp requires shape (2, ), but get {amp.shape}")
        try:
            amp = amp / np.sqrt(np.vdot(amp, amp))
        except ZeroDivisionError as exc:
            raise ZeroDivisionError("Mode of amp is zero.") from exc
        global_phase = np.angle(amp[0])
        amp = amp / np.exp(1j * global_phase)
        theta = 2 * np.arccos(np.real(amp[0]))
        phi = np.angle(amp[1])
        x, y, z = np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)
        return np.array([x, y, z])

    # pylint: disable=too-many-arguments,too-many-locals,too-many-branches,too-many-statements
    def add_state(
        self,
        ax,
        amp,
        linecolor=None,
        linewidth=None,
        pointcolor=None,
        pointsize=None,
        marker=None,
        projcolor=None,
        mode=None,
        with_proj=None,
        stick_args=None,
        stick_kwargs=None,
        point_args=None,
        point_kwargs=None,
        proj_args=None,
        proj_kwargs=None,
    ):
        """
        Add one quantum state on bloch sphere.

        Args:
            ax (mpl_toolkits.mplot3d.axes3d.Axes3D): The three dimension axes you
                want add quantum state.
            amp (numpy.ndarray): The quantum state.
            linecolor (str): The color for stick. If ``None``, it will be found in `config`
                in `BlochScene` with key `stick_c`. Default: ``None``.
            linewidth (numbers.Number): The line width for stick. If ``None``, it will be found
                in `config` in `BlochScene` with `stick_w`. Default: ``None``.
            pointcolor (str): The color for point. If ``None``, it will be found in `config`
                in `BlochScene` with key `point_c`. Default: ``None``.
            pointsize (numbers.Number): The size of point. If ``None`` it will be found in `config`
                in `BlochScene` with key `point_s`. Default: ``None``.
            marker (str): Point marker. If ``None``, it will be found in `config` in `BlochScene` with
                key `point_m`. Default: ``None``.
            projcolor (str): Project line color. If ``None`` it will be found in `config` in `BlochScene`
                with key `proj_c`. Default: ``None``.
            mode (str): How to display the quantum state. Can be one of 'stick', 'point', 'both'.
                If ``None``, if will be found in `config` of `BlochScene` with key `state_mode`.
                Default: ``None``.
            with_proj (bool): Whether to display the projection line alone x, y and z axis. If ``None``,
                it will be found in `config` in `BlochScene` with key `with_proj`. Default: ``None``.
            stick_args (tuple): The other args for stick. These args will be send to `Axes3D.plot`.
                Default: ``None``.
            stick_kwargs (dict): The other key word args for stick. These args will be send to `Axes3D.plot`.
                Default: ``None``.
            point_args (tuple): The other args for point. These args will be send to `Axes3D.scatter`.
                Default: ``None``.
            point_kwargs (dict): The other key word args for point. These args will be send to
                `Axes3D.scatter`. Default: ``None``.
            proj_args (tuple): The other args for projection line. These args will be send
                to `Axes3D.plot`. Default: ``None``.
            proj_kwargs (dict): The other key word args for projection line. These args will be send to
                `Axes3D.plot`. Default: ``None``.

        Returns:
            dict, a dict of object of stick, point, and projection line.
        """
        if linecolor is None:
            linecolor = self.config['stick_c']
        _check_input_type('linecolor', str, linecolor)
        if linewidth is None:
            linewidth = self.config['stick_w']
        _check_input_type('linewidth', numbers.Number, linewidth)
        if pointcolor is None:
            pointcolor = self.config['point_c']
        _check_input_type('pointcolor', str, pointcolor)
        if pointsize is None:
            pointsize = self.config['point_s']
        _check_input_type('pointsize', numbers.Number, pointsize)
        if marker is None:
            marker = self.config['point_m']
        _check_input_type('marker', str, marker)
        if projcolor is None:
            projcolor = self.config['proj_c']
        _check_input_type('projcolor', str, projcolor)
        if mode is None:
            mode = self.config['state_mode']
        if mode not in ('both', 'stick', 'point'):
            raise ValueError(f"mode should be one of 'both', 'stick' or 'point'. But get {mode}")
        if with_proj is None:
            with_proj = self.config['with_proj']
        _check_input_type('with_proj', bool, with_proj)
        if stick_args is None:
            stick_args = ()
        _check_input_type('stick_args', tuple, stick_args)
        if point_args is None:
            point_args = ()
        _check_input_type('point_args', tuple, point_args)
        if proj_args is None:
            proj_args = ()
        _check_input_type('proj_args', tuple, proj_args)
        if stick_kwargs is None:
            stick_kwargs = {}
        _check_input_type('stick_kwargs', dict, stick_kwargs)
        if point_kwargs is None:
            point_kwargs = {}
        _check_input_type('point_kwargs', dict, point_kwargs)
        if proj_kwargs is None:
            proj_kwargs = {}
        _check_input_type('proj_kwargs', dict, proj_kwargs)

        x, y, z = self.state_to_cor(amp)
        objs = {}
        if with_proj:
            line_x = ax.plot([0, x], [0, 0], [0, 0], projcolor, *proj_args, linewidth=linewidth, **proj_kwargs)[0]
            line_y = ax.plot([0, 0], [0, y], [0, 0], projcolor, *proj_args, linewidth=linewidth, **proj_kwargs)[0]
            line_z = ax.plot([0, 0], [0, 0], [0, z], projcolor, *proj_args, linewidth=linewidth, **proj_kwargs)[0]
            objs['line_x'] = line_x
            objs['line_y'] = line_y
            objs['line_z'] = line_z
        if mode in ('both', 'stick'):
            vec = ax.plot([0, x], [0, y], [0, z], linecolor, *stick_args, linewidth=linewidth, **stick_kwargs)[0]
            objs['vec'] = vec
        if mode in ('both', 'dot'):
            point = ax.scatter([x], [y], [z], *point_args, s=pointsize, marker=marker, c=pointcolor, **point_kwargs)
            objs['point'] = point
        return objs

    def update(self, objs: dict, new_amp: np.ndarray):
        """
        Update quantum state in bloch sphere.

        Update the quantum state in bloch sphere for a given objs generated
        by `BlochScene.add_state` and a given quantum state.

        Args:
            objs (dict): The objects generated by `BlochScene.add_state`.
            new_amp (numpy.ndarray): The new quantum state.
        """
        x, y, z = self.state_to_cor(new_amp)
        if 'vec' in objs:
            vec = objs['vec']
            vec.set_data(np.array([[0, x], [0, y]]))
            vec.set_3d_properties(np.array([0, z]))
        if 'point' in objs:
            point = objs['point']
            point._offsets3d = np.array([[x, y, z]]).T  # pylint: disable=protected-access
        if 'line_x' in objs:
            line_x = objs['line_x']
            line_x.set_data(np.array([[0, x], [0, 0]]))
            line_x.set_3d_properties(np.array([0, 0]))
        if 'line_y' in objs:
            line_y = objs['line_y']
            line_y.set_data(np.array([[0, 0], [0, y]]))
            line_y.set_3d_properties(np.array([0, 0]))
        if 'line_z' in objs:
            line_z = objs['line_z']
            line_z.set_data(np.array([[0, 0], [0, 0]]))
            line_z.set_3d_properties(np.array([0, z]))

    # pylint: disable=too-many-arguments,too-many-locals
    def animation(self, fig, ax, objs, new_amps: np.ndarray, interval=15, with_trace=True, history_len=None, **kwargs):
        """
        Animate a quantum state on bolch sphere.

        Args:
            fig (matplotlib.figure.Figure): The bloch sphere scene figure.
            ax (mpl_toolkits.mplot3d.axes3d.Axes3D): The axes of bloch sphere scene.
            objs (dict): The objects generated by `BlochScene.add_state`.
            new_amps (numpy.ndarray): All quantum state you want to animate.
            interval (int): Delay between frames in milliseconds. Default: ``15``.
            with_trace (bool): Whether to display the trace of quantum state. Default: ``True``.
            history_len (int): The trace length. If ``None``, it will be the length of given
                quantum states. Defaults: ``None``.
            kwargs (dict): The other key word args for `animation.FuncAnimation`.

        Returns:
            animation.FuncAnimation, the animation object in matplotlib.
        """
        console = rich.console.Console()
        if console.is_jupyter:
            warnings.warn(
                "jupyter environment detected, if animation not work, please install ipympl with "
                "'!pip install ipympl' in jupyter notebook and run '%matplotlib ipympl' in cell."
            )
        _check_input_type('fig', Figure, fig)
        _check_input_type('ax', Axes3D, ax)
        _check_input_type('objs', dict, objs)
        _check_input_type('new_amps', np.ndarray, new_amps)
        _check_int_type('interval', interval)
        _check_input_type('with_trace', bool, with_trace)
        amps_shape = new_amps.shape
        if len(amps_shape) != 2 or amps_shape[1] != 2:
            raise ValueError(f"new_amps requires two dimension with second dimension size to 2, but get {amps_shape}")
        if with_trace:
            trace_color = self.config['trace_c']
            trace_marker = self.config['trace_m']
            trace_lw = self.config['trace_lw']
            trace_ms = self.config['trace_ms']
            trace_ls = self.config['trace_ls']
            (trace,) = ax.plot(
                [], [], [], color=trace_color, linestyle=trace_ls, marker=trace_marker, lw=trace_lw, ms=trace_ms
            )
            if history_len is None:
                history_len = new_amps.shape[0]
            _check_int_type('history_len', history_len)
            his_x, his_y, his_z = deque(maxlen=history_len), deque(maxlen=history_len), deque(maxlen=history_len)

            def func(i):
                """Update func with projection lines."""
                x, y, z = self.state_to_cor(new_amps[i])
                if i == 0:
                    his_x.clear()
                    his_y.clear()
                    his_z.clear()
                his_x.appendleft(x)
                his_y.appendleft(y)
                his_z.appendleft(z)
                self.update(objs, new_amps[i])
                trace.set_data(np.array([his_x, his_y]))
                trace.set_3d_properties(np.array(his_z))

        else:

            def func(i):
                self.update(objs, new_amps[i])

        return animation.FuncAnimation(fig, func, new_amps.shape[0], interval=interval, **kwargs)

def plot_state_bloch(quantum_state):
    scene = BlochScene()
    _, ax = scene.create_scene()
    scene.add_state(ax, quantum_state)
    return scene
