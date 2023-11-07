#!/bin/bash
cd ../
python quick_test.py --config_file_path ./config.yaml --device_target CPU --device_id 0 --mode PYNATIVE --save_graphs False --save_graphs_path ./graphs