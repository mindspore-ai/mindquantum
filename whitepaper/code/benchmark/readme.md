# Benchmark

## system info

|System|Info|
|-|-|
|Python|3.9.5|
|OS|Linux x86_64|
|Memory|16.62 GB|
|CPU Max Thread|16|

## Install package

Create isolated conda environment.

```bash
conda create -n benchmark python=3.9.5
conda activate benchmark
```

### Install Requirements

```bash
pip3 install -r requirements.txt
pip3 install protobuf==3.20.1
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

### Install Intel

Clone and build intel master

```bash
CUR_DIR=`pwd`
pip3 install pybind11
git clone https://github.com/intel/intel-qs.git
cd intel-qs
mkdir -p build
cd build
cmake -DIqsPython=ON ..
make -j8
cp ./lib/intelqs_py*.so `python -c "import site;print(site.getsitepackages()[0])"`
cd $CUR_DIR
```

Force use CPU

```bash
export CUDA_VISIBLE_DEVICES=""
```

Total Thread 16

```
export OMP_NUM_THREADS=16
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
