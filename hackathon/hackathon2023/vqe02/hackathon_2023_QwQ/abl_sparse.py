from eval import *
from src.ocvqe import *

# 导出 PPT 用对比表：QP-uccsd-trotter 权重稀疏化

trotter = 4
enable_sparse = True
sparse_thresh = 3e-4
enable_export = True
ckk_step = 0
chk_interval = 100
weight_hists = []

if enable_export:
  import matplotlib.pyplot as plt
  from moviepy.editor import ImageSequenceClip


def scipy_callback_fn(xk:ndarray) -> bool:
  global ckk_step, chk_interval
  ckk_step += 1
  if ckk_step % chk_interval != 0: return False
  print(f'>> step: {ckk_step}')

  global enable_export, weight_hists
  if enable_export:
    plt.subplot(121) ; plt.title('w')      ; plt.hist(xk, bins=50)
    plt.subplot(122) ; plt.title('log|w|') ; plt.hist(np.log(np.abs(xk) + 1e-9), bins=50)
    plt.suptitle(f'weight hist of step: {ckk_step}')
    cvs = plt.gcf().canvas
    cvs.draw()
    img = np.frombuffer(cvs.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(cvs.get_width_height()[::-1] + (3,))
    weight_hists.append(img)
    plt.clf()
    plt.close()

  if enable_sparse:
    mask = np.abs(xk) > sparse_thresh
    xk *= mask     # inplace

  return False

def ocvqe_solver(mol:MolecularData, config1:Config, config2:Config) -> float:
  ham = get_ham(mol, config1)

  # Ground state E0: |ψ(λ0)>
  gs_sim, gs_ene, params = run_gs(mol, ham, config1)

  # NOTE: setup scipy.optim callback
  from src import common
  common.scipy_callback = scipy_callback_fn

  # Excited state E1: |ψ(λ1)>
  es_ene, params = run_es(mol, ham, gs_sim, config2, params)

  return es_ene

def run_ocvqe(mol:MolecularData) -> float:
  config1 = {
    'ansatz':  'UCCSD-QP-hijack',
    'trotter': 1,
    'optim':   'BFGS',
    'tol':     1e-4,
    'maxiter': 100,
    'dump':    False,
  }
  config2 = {
    'ansatz':  'UCCSD-QP-hijack',
    'trotter': trotter,
    'optim':   'BFGS',
    'tol':     1e-8,
    'beta':    4,
    'eps':     1e-5,
    'maxiter': 1000,
    'cont_evolve': False,
  }
  return ocvqe_solver(mol, config1, config2)


for idx, (name, E0_gt, E1_gt) in enumerate(molecules):
  print(f'[{name}]')

  mol = MolecularData(filename=os.path.join(BASE_PATH, f'./molecule_files/{name}'))
  mol.load()

  seed_everything(42)

  t = time()
  E1_hat = run_ocvqe(mol)
  print(f'[trotter_{trotter}_sparse] error: {abs(E1_hat - E1_gt)}, time: {time() - t}')

  # NOTE: export video
  if enable_export and weight_hists:
    clips = [weight_hists[0]] * 1 + weight_hists + [weight_hists[-1]] * 4
    clip = ImageSequenceClip(clips, fps=1)
    clip.write_videofile('abl_sparse.mp4', fps=1)
