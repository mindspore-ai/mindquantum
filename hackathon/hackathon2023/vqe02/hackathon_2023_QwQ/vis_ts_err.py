#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/08/07

import zipfile as zf
from pprint import pprint as pp
from argparse import ArgumentParser

import matplotlib.pyplot as plt

from eval_H2O import *
thresh = 0.0016

RESULTS_PATH: Path = BASE_PATH / 'results'


def collect():
  hist = []

  # results fro HiQ platform
  fp = RESULTS_PATH / 'eval_H2O.json'
  if fp.exists():
    with open(fp, 'r', encoding='utf-8') as fh:
      db = json.load(fh)
      hist.extend(db['hist'])

  # results fro HiQ cluster
  for fp in RESULTS_PATH.iterdir():
    if not fp.suffix == '.zip': continue

    with zf.ZipFile(fp) as fh:
      info = fh.getinfo('result/runSpace/HiQJob/hackathon_vqe_02/eval_H2O.json')
      with fh.open(info) as fh2:
        db = json.load(fh2)
        hist.extend(db['hist'])

  print(f'>> collect: {len(hist)} items')
  save_db({'hist': hist}, DB_FILE)


def vis():
  db = load_db(DB_FILE)

  tss, errs, pair = [], [], []
  for rec in db['hist']:
    if rec['err'] > 10 * thresh: continue
    if rec['ts'] > 40: continue
    tss .append(rec['ts'])
    errs.append(rec['err'])
    pair.append((rec['err'], rec['ts'], rec['seed']))
  
  pair.sort()
  pp(pair[:50])

  plt.scatter(tss, errs, alpha=0.7, s=3)
  plt.savefig(DB_FILE.with_suffix('.png'), dpi=600)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--collect', action='store_true', help='collect results to make db json')
  args = parser.parse_args()

  if args.collect: collect()
  vis()
