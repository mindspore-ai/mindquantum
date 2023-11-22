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
pip3 install pytest pytest-benchmark
```

Install mindquantum 0.9.0

```bash
pip3 install mindquantum==0.9.0
```

Install qulacs 0.6.2:

```bash
pip3 install qulacs==0.6.2
```

Install qiskit-aer 0.13.0

```bash
pip3 install qiskit-aer==0.13.0
```

Install projectq 0.8.0

```bash
pip3 install projectq==0.8.0
```

Install pennylane 0.33.1

```bash
pip3 install pennylane==0.33.0
```

Clone and build quest v3.7.0

```bash
CUR_DIR=`pwd`
git clone --branch v3.7.0 --single-branch https://github.com/QuEST-Kit/QuEST.git
cd QuEST
mkdir -p build
cd build
cmake ..
make -j8
cd $CUR_DIR
```

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

Install pyqpanda 3.8.0

```bash
pip3 install pyqpanda==3.8.0
```

Install paddle quantum 2.4.0

```bash
pip3 install paddle-quantum==2.4.0
```

Install tensorcircuit 0.11.0

```bash
pip3 install tensorcircuit==0.11.0
```

Install tensorflow quantum 0.7.2

```bash
pip3 install tensorflow-quantum==0.7.2
```

Install other packages.

```bash
pip3 install networkx
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
