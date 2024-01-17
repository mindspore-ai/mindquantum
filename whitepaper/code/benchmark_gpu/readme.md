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
```

### Install Requirements

```bash
pip install -r requirements.txt
pip install protobuf==3.20.1
```

### Install Qulacs

```bash
CUR_DIR=`pwd`
wget https://files.pythonhosted.org/packages/23/a2/6fca76d22de5f85ed3707cfeb47b09cc5b4a5edaf7af930b32f17257fa95/Qulacs-GPU-0.3.1.tar.gz
tar -xvf Qulacs-GPU-0.3.1.tar.gz
cd Qulacs-GPU-0.3.1
sed -i '17c\set(DEFAULT_USE_TEST No)' CMakeLists.txt
python3 setup.py install
cd $CUR_DIR
```

### Install PyQpanda

```bash
CUR_DIR=`pwd`
git clone https://gitee.com/donghufeng/QPanda-2
cd QPanda-2
sed -i '4c\cmake -DFIND_CUDA=ON -DUSE_CHEMIQ=OFF -DUSE_PYQPANDA=ON ..' build.sh
sed -i '5c\make -j16' build.sh
bash build.sh
export PYTHONPATH=`pwd`/pyQPanda:$PYTHONPATH
```

## Set Environment

```bash
site_packages=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH=$site_packages/cuquantum/lib:$site_packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
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

Or run with a script

```bash
. ./run.sh
```

Take a look at the previous result.

```bash
pytest-benchmark compare 0009
```
