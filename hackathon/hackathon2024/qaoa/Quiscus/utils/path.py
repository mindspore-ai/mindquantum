#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/05/31 

from pathlib import Path

BASE_PATH = Path(__file__).parent.parent
LOG_PATH = BASE_PATH / 'log' ; LOG_PATH.mkdir(exist_ok=True)
IMG_PATH = BASE_PATH / 'img'
DATA_OPT_PATH = BASE_PATH / 'data_opt'

LOOKUP_FILE = BASE_PATH / 'utils' / 'transfer_data.csv'
