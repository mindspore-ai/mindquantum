#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/08/23

from vis_xyz import *
from tqdm import tqdm
from moviepy.editor import ImageSequenceClip

# 导出 PPT 用 demo 视频


def draw3D(ene, geo):
  T = geo.shape[0]
  imgs = []
  for i in tqdm(range(T)):
    E = ene[i]
    xyz = geo[i, :, :]  # [N, D]

    fig, ax = plt.subplots()
    ax = plt.axes(projection='3d')
    ax._axis3don = False
    ax.scatter3D(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=50)
    fig.suptitle(f'predict energy: {E}')
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    imgs.append(img)
    ax.clear()
    fig.clear()
    plt.close()
  return imgs

def draw2D(ene, geo):
  T, N, D = geo.shape
  
  if 'pca':
    from sklearn.decomposition import PCA
    geo_flat = geo.reshape([T*N, D])
    pca = PCA(n_components=2)
    geo_pca = pca.fit_transform(geo_flat)
    geo = geo_pca.reshape([T, N, 2])

  imgs = []
  for i in tqdm(range(T)):
    E = ene[i]
    xy = geo[i, :, :]  # [N, D]

    fig, ax = plt.subplots()
    ax.scatter(xy[:, 0], xy[:, 1], s=50)
    ax.axis('off')
    fig.suptitle(f'predict energy: {E}')
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    imgs.append(img)
    ax.clear()
    fig.clear()
    plt.close()
  return imgs


def export(args):
  with open(args.f, 'r', encoding='utf-8') as fh:
    data = json.load(fh)
  pp(data['args'])

  ene = np.asarray(data['energy'])      # [T]
  geo = np.asarray(data['geometry'])    # [T, N, D]

  if args._3d:
    imgs = draw3D(ene, geo)
  else:
    imgs = draw2D(ene, geo)
  
  clip = ImageSequenceClip(imgs, fps=args.fps)
  clip.write_videofile('anim.mp4', fps=args.fps)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-f', required=True, type=Path, help='track file stats.json')
  parser.add_argument('--fps', default=15, type=float, help='export video FPS')
  parser.add_argument('--_3d', action='store_true', help='use plt.scatter3D')
  args = parser.parse_args()

  f: Path = Path(args.f)
  if f.is_dir(): args.f = f = f / 'stats.json'
  assert f.is_file() and f.suffix == '.json'

  export(args)
