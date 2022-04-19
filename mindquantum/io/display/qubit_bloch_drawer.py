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

__all__ = ['decompose', 'draw_single_qubit_bloch',
           'draw_single_qubit_bloch_time_evolution',
           'draw_on_jupyter',]

import io
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from .bloch import SQBloch
from ._config import _bloch_drawer_config
ipythonisavailable = True
try:
    from IPython import display
except ImportError:
    ipythonisavailable = False

def _decompose(amp):
    """Decompose amplitude to coordinate."""
    rh = (amp[0].real ** 2 + amp[0].imag ** 2) ** 0.5
    if rh > 1e-8:
        amp /= amp[0] / rh # Global phase
    theta = np.arccos(amp[0].real) * 2
    z = (1 - amp[0].real ** 2) ** 0.5
    phi = np.log(amp[1] / z).imag if z > 1e-8 else 0
    return [np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)]

def decompose(amplitudes):
    """
    Decompose amplitudes to coordinate.

    Args:
        amplitude (np.ndarray): Quantum states of time evolving single qubit.

    Returns:
        vs (list): Three dimensional coordinate representation of time evolution qubits on Bloch sphere.

    Raises:
        ValueError: If `amplitude` is not a `np.ndarray`.
        ValueError: If the size of `amplitude` is inappropriate.

    Examples:
        >>> import numpy as np
        >>> from mindquantum.io.display import decompose
        >>> amp = np.array([-0.29426025+0.64297038j,
                            -0.29426025-0.64297038j])
        >>> decompose(amp)
        >>> amps = np.array([[-0.29426025+0.64297038j,
                              -0.29426025-0.64297038j]] * 10)
        >>> decompose(amps)
    """
    if not isinstance(amplitudes, np.ndarray):
        raise ValueError("Type {} is not supported".format(type(amplitudes)))
    amplitudes = amplitudes.squeeze()
    s = amplitudes.shape
    try:
        if s[-1] != 2 or len(s)>2:
            raise ValueError("Invalid amplitudes!")
    except TypeError:
        raise ValueError("Invalid amplitudes!")
    except IndexError:
        raise ValueError("Invalid amplitudes!")

    if len(amplitudes.shape) == 1:
        return _decompose(amplitudes)
    vs = []
    for am in amplitudes:
        vs.append(_decompose(am))
    return vs

def draw_single_qubit_bloch(amplitude, ax=None, fig=None, figsize=(8, 8), dpi=100, **kwargs):
    """
    Draw the quantum state on bloch.

    Args:
        amplitude (np.ndarray): Quantum state of single qubit.
        ax (matplotlib.axes._subplots.Axes3DSubplot or None): 3D canvas. Default: None.
        fig (matplotlib.figure.Figure or None): Figure. Default: None.
        figsize (tuple or list): The size of figure. Default: (8, 8).
        dpi (int): The dpi of figure. Default: 100.
        **kwargs : See more from `.bloch.SQBloch.__init__`.

    Returns:
        fig (matplotlib.figure.Figure or None): Figure.
        ax (matplotlib.axes._subplots.Axes3DSubplot or None): 3D canvas.

    Raises:
        ValueError: If `amplitude` is not a `np.ndarray`.
        ValueError: If the size of `amplitude` is inappropriate.

    Examples:
        >>> import numpy as np
        >>> from mindquantum.io.display import draw_single_qubit_bloch
        >>> amp = np.array([-0.29426025+0.64297038j,
                           -0.29426025-0.64297038j])
        >>> fig, ax = draw_single_qubit_bloch(amp, label=['x', 'y', ['$\\left|0\\right>$', '$\\left|1\\right>$']])
    """
    if ax is None:
        if fig is None:
            fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(projection="3d")
    vec = decompose(amplitude)
    b = SQBloch(kwargs, **_bloch_drawer_config)
    b.plot(ax, vec)
    return fig, ax

def draw_single_qubit_bloch_time_evolution(amplitudes,
                                           ax=None, fig=None,
                                           figsize=(8, 8), dpi=100,
                                           interval=0.3, filename=None,
                                           **kwargs):
    """
    Draw the quantum states of time evolving single qubit on bloch.

    Args:
        amplitude (np.ndarray): Quantum states of time evolving single qubit.
        ax (matplotlib.axes._subplots.Axes3DSubplot or None): 3D canvas. Default: None.
        fig (matplotlib.figure.Figure or None): Figure. Default: None.
        figsize (tuple or list): The size of figure. Default: (8, 8).
        dpi (int): The dpi of figure. Default: 100.
        interval (float): Interval per frame. Default: 0.3.
        filename (str or None): `*.gif` if you want to savefig. Default: None.
        **kwargs : See more from `.bloch.SQBloch.__init__`.

    Returns:
        fig (matplotlib.figure.Figure or None): Figure.
        ax (matplotlib.axes._subplots.Axes3DSubplot or None): 3D canvas.
        ims (list): Frames list.

    Raises:
        ValueError: If `amplitude` is not a `np.ndarray`.
        ValueError: If the size of `amplitude` is inappropriate.

    Examples:
        >>> import numpy as np
        >>> from mindquantum.io.display import draw_single_qubit_bloch_time_evolution
        >>> amp = np.array([[1, 0], [0, 1]] * 3)
        >>> fig, ax, ims = draw_single_qubit_bloch_time_evolution(amps)
    """
    savef = isinstance(filename, str)
    if ax is None:
        if fig is None:
            fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(projection="3d")
    if fig is None:
        raise RuntimeError("You need to set fig.")
    fsize = fig.bbox_inches.get_points()[1]*fig.dpi
    fsize = (int(fsize[0]), int(fsize[1]))
    vecs = decompose(amplitudes)
    b = SQBloch(kwargs, **_bloch_drawer_config)
    ims = []

    for v in vecs:
        buffer_ = io.BytesIO()
        ax.cla()
        b.plot(ax, v)
        fig.savefig(buffer_, format = "png")
        buffer_.seek(0)
        img = Image.open(buffer_)
        ims.append(img.resize(fsize, Image.ANTIALIAS))
        buffer_.close()
        plt.pause(interval)

    if savef:
        ims[0].save(filename, save_all=True, append_images=ims[1:], duration=interval * 1000)
    return fig, ax, ims

def jupyterfflag(fable):
    def outwrapper(func):
        def wrapper(images, interval, f=False):
            f = fable
            return func(images, interval, f)
        return wrapper
    return outwrapper

@jupyterfflag(fable=ipythonisavailable)
def draw_on_jupyter(images, interval, fable=False):
    """
    Draw the dynamic graph of bloch on the jupyter notebook with matplotlib inline.

    Examples:
        In [ ]: from mindquantum.io.display import draw_single_qubit_bloch_time_evolution
        In [ ]: from mindquantum.io.display import draw_on_jupyter
        In [ ]: %matplotlib inline
        In [ ]: amps = [your amplitude list]
        In [ ]: fig, ax, ims = draw_single_qubit_bloch_time_evolution(amps)
        In [ ]: draw_on_jupyter(ims, 0.1)
    """
    for im in images:
        plt.axis("off")
        plt.imshow(im)
        if fable:
            display.clear_output(wait=True)
        plt.pause(interval)
    plt.show()

def _main(graph=0):
    """
    Test.
        'draw_single_qubit_bloch' if graph == 0
        'draw_single_qubit_bloch_time_evolution' if graph == 1
        'draw_single_qubit_bloch_time_evolution_static' if graph == 2
    """
    if not graph:
        amp = np.array([-0.29426025+0.64297038j,
                        -0.29426025-0.64297038j])
        v = decompose(amp)
        print(v)
        draw_single_qubit_bloch(amp, label=['x', 'y', ['$\\left|0\\right>$', '$\\left|1\\right>$']])
        plt.show()
    else:
        w = np.matrix([[1, 0]])
        h = 1. / np.sqrt(2.) * np.matrix([[1, 1], [1, -1]])
        ang = np.pi / 8
        rz = np.matrix([[np.exp(-.5 * 1j * ang), 0],
                        [0, np.exp(.5 * 1j * ang)]])
        amps = []
        amps.append(w)
        w = w * h
        amps.append(w)
        for _ in range(18):
            w = w * rz
            amps.append(w)
        amps = np.array(amps)

        fig = plt.figure(figsize=(8,8))
        fig, _, ims = draw_single_qubit_bloch_time_evolution(amps, fig=fig, interval=0.1,
                                                             filename='bloch.gif',
                                                             ax_offset=(10, 35),
                                                             )
        plt.show()
        jupyter = False
        if jupyter:
            draw_on_jupyter(ims, 0.1)

if __name__ == '__main__':
    _main(2)
