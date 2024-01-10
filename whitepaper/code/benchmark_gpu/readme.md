# Benchmark

## system info

|System|Info|
|-|-|
|Python|3.9.5|
|OS|Linux x86_64|
|Memory|16.62 GB|
|GPU|RTX 2060|

## Install package

Create isolated conda environment.

```bash
conda create -n benchmark_gpu python=3.9.5
conda activate benchmark_gpu
pip3 install pytest pytest-benchmark
```

Install mindquantum 0.9.0

```bash
pip3 install mindquantum==0.9.0
```

Install qulacs-gpu 0.3.1:

```bash
pip3 install qulacs-gpu==0.3.1
```

Install qiskit-aer-gpu 0.12.2

```bash
pip3 install qiskit-aer-gpu==0.12.2
```

Install pennylane 0.33.0

```bash
pip3 install pennylane-lightning-gpu==0.33.0
pip3 install pennylane-qiskit==0.33.1
site_packages=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH=$site_packages/cuquantum/lib:$site_packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
```

Install pyqpanda 3.8.0

```bash
pip3 install pyqpanda==3.8.0
```

Install tensorcircuit 0.11.0

```bash
pip3 install tensorcircuit==0.11.0
```

Install tensorflow quantum 0.7.2

```bash
pip3 install tensorflow==2.7
pip3 install tensorflow-quantum==0.7.2
```

Install other packages.

```bash
pip3 install networkx
pip3 install protobuf==3.20.1
```

## Generate dataset

```bash
CUR_DIR=`pwd`
cd utils
python3 generate_random_circuit.py
python3 generate_graph.py
cd $CUR_DIR
```


## Run benchmark

```bash
cd src
pytest -v --benchmark-save=all --benchmark-warmup=on --benchmark-warmup-iterations=1
```

Or only benchmark one frame with one task, for example

```bash
pytest -v --benchmark-save=mindquantum --benchmark-warmup=on --benchmark-warmup-iterations=1 -m 'random_circuit and mindquantum'
```

Take a look at the previous result.

```bash
pytest-benchmark compare 0009
```
