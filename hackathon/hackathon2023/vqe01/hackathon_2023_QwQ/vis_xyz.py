#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/07/14

import json
from pathlib import Path
from pprint import pprint as pp
from argparse import ArgumentParser
from traceback import print_exc

import tkinter as tk
import tkinter.ttk as ttk
import numpy as np
import matplotlib as mpl ; mpl.rcParams['figure.figsize'] = [8, 10]
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# visualize geometry of the moleculer


class App:

  def __init__(self, args):
    with open(args.f, 'r', encoding='utf-8') as fh:
      data = json.load(fh)
    pp(data['args'])

    self.ene = np.asarray(data['energy'])      # [T]
    self.geo = np.asarray(data['geometry'])    # [T, N, D]

    self.setup_gui()

    try:
      self.wnd.mainloop()
    except KeyboardInterrupt:
      self.wnd.quit()
    except: print_exc()
    
  def setup_gui(self):
    # window
    wnd = tk.Tk()
    wnd.title('Moleculer Structure Visualizer')
    wnd.protocol('WM_DELETE_WINDOW', wnd.quit)
    self.wnd = wnd

    # top: query
    frm1 = ttk.Label(wnd)
    frm1.pack(side=tk.TOP, anchor=tk.N, expand=tk.YES, fill=tk.X)
    if True:
      self.var_step = tk.IntVar(frm1, value=0)
      tk.Label(frm1, text='Steps').pack(side=tk.LEFT, expand=tk.NO)
      sc = tk.Scale(frm1, command=lambda _: self.refresh(), variable=self.var_step, 
                    orient=tk.HORIZONTAL, from_=0, to=self.geo.shape[0]-1, 
                    tickinterval=100, resolution=1)
      sc.pack(expand=tk.YES, fill=tk.X)
      self.sc = sc

    # bottom: plot
    frm2 = ttk.Frame(wnd)
    frm2.pack(side=tk.BOTTOM, expand=tk.YES, fill=tk.BOTH)
    if True:
      ax = plt.axes(projection='3d')
      cvs = FigureCanvasTkAgg(ax.figure, frm2)
      cvs.get_tk_widget().pack(expand=tk.YES, fill=tk.BOTH)
      self.ax, self.cvs = ax, cvs
      self.refresh()

  def refresh(self):
    i = self.var_step.get()

    E = self.ene[i]
    xyz = self.geo[i, :, :]  # [N, D]

    self.ax.clear()
    self.ax.scatter3D(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    self.ax.set_title(f'predict energy: {E}')
    self.cvs.draw()


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-f', required=True, type=Path, help='track file stats.json')
  args = parser.parse_args()

  f: Path = Path(args.f)
  if f.is_dir(): args.f = f = f / 'stats.json'
  assert f.is_file() and f.suffix == '.json'

  App(args)
