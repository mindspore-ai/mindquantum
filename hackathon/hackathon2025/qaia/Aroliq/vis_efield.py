#!/usr/bin/env python3
# Author: Armit
# Create Time: 周六 2025/05/10

# 该任务的基本思想就是用若干个钟形辐射场叠加出一个目标形状辐射场

from argparse import ArgumentParser
from traceback import format_exc, print_exc
from typing import List

import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox as tkmsg

import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# 生成单元阵子的辐射电场强度（随角度变化的函数）
def generate_power_pattern(n_angle:int) -> ndarray:
  theta = np.linspace(0, 180, 180 * n_angle + 1)
  x = 12 * ((theta - 90) / 90) ** 2
  E_dB = -1.0 * np.where(x < 30, x, 30)
  E_theta = 10 ** (E_dB / 10)
  EF = E_theta ** 0.5
  return EF

# Eq. 9 阵因子A_n
def generate_array_factor(N:int, n_angle:int):
  theta = np.linspace(0, 180, 180 * n_angle + 1)
  phase_x = 1j * np.pi * np.cos(theta * np.pi / 180)
  AF = np.exp(phase_x[None, :] * np.arange(N)[:, None])
  return AF

# 调制后的场
def get_BF(efield:ndarray, phase_angle:ndarray, amp:ndarray=1.0):
  F = np.einsum('i, ij -> j', amp * np.exp(1.0j * phase_angle), efield)
  FF = np.real(np.conj(F) * F)
  FF_n = (FF / np.max(FF)) if np.max(FF) > 0 else 1e-5
  y = 10 * np.log10(FF_n)
  return y


class App:

  def __init__(self, args):
    self.N: int = args.N
    self.M: int = args.M
    self.vars_phi: List[tk.DoubleVar] = [None] * self.N
    self.vars_amp: List[tk.DoubleVar] = [None] * self.N

    EF = generate_power_pattern(self.M)         # [D=1801]，每个天线阵子独立产生的一个辐射场(像个钟形曲线的概率分布列)
    AF = generate_array_factor(self.N, self.M)  # [N=4, D=1801], N天线阵子数，D各方位角度细分数
    self.efield = EF[None, ...] * AF
    self.init_phi = 0.0
    self.init_amp = 0.5

    self.setup_gui()

    try:
      self.wnd.mainloop()
    except KeyboardInterrupt:
      self.wnd.quit()
    except: print_exc()

  def setup_gui(self):
    # window
    wnd = tk.Tk()
    wnd.title('Visualize Encoder')
    wnd.protocol('WM_DELETE_WINDOW', wnd.quit)
    self.wnd = wnd

    # top: plot
    frm1 = ttk.Label(wnd)
    frm1.pack(side=tk.TOP, anchor=tk.N, expand=tk.YES, fill=tk.X)
    if True:
      fig, ax = plt.subplots()
      fig.tight_layout()
      cvs = FigureCanvasTkAgg(fig, frm1)
      cvs.get_tk_widget().pack(expand=tk.YES, fill=tk.BOTH)
      self.fig, self.ax, self.cvs = fig, ax, cvs

    # bottom: controls
    frm2 = ttk.Frame(wnd)
    frm2.pack(side=tk.BOTTOM, expand=tk.YES, fill=tk.BOTH)
    if True:
      for i in range(self.N):
        self.vars_phi[i] = tk.DoubleVar(frm2, value=self.init_phi)
        self.vars_amp[i] = tk.DoubleVar(frm2, value=self.init_amp)

        frm2x = ttk.Label(frm2)
        frm2x.pack(side=tk.LEFT, expand=tk.YES, fill=tk.Y)
        if True:
          frm2x1 = ttk.Label(frm2x)
          frm2x1.pack(expand=tk.YES, fill=tk.Y)
          if True:
            tk.Label(frm2x1, text=f'phi-{i}').pack(side=tk.TOP, expand=tk.NO)
            tk.Scale(frm2x1, command=lambda _: self.redraw(), variable=self.vars_phi[i], orient=tk.VERTICAL, from_=2*np.pi, to=0, resolution=0.01).pack(expand=tk.YES, fill=tk.Y)

          frm2x2 = ttk.Label(frm2x)
          frm2x2.pack(expand=tk.YES, fill=tk.Y)
          if True:
            tk.Label(frm2x2, text=f'amp-{i}').pack(side=tk.TOP, expand=tk.NO)
            tk.Scale(frm2x2, command=lambda _: self.redraw(), variable=self.vars_amp[i], orient=tk.VERTICAL, from_=1, to=0, resolution=0.01).pack(expand=tk.YES, fill=tk.Y)

    self.redraw()

  def redraw(self):
    try:
      phi = np.asarray([v.get() for v in self.vars_phi], dtype=np.float32)
      amp = np.asarray([v.get() for v in self.vars_amp], dtype=np.float32)
      y = get_BF(self.efield, phi, amp)

      self.ax.cla()
      self.ax.plot(y)
      self.cvs.draw()
    except:
      info = format_exc()
      print(info)
      tkmsg.showerror('Error', info)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-N', default=6, type=int, help='number of antenna site')
  parser.add_argument('-M', default=10, type=int, help='number of divisions for 1°')
  args = parser.parse_args()

  App(args)
