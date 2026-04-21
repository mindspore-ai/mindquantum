`spin_systems` is not stored in this repository.

Download the local TensorFlow Quantum TFI chain dataset into this directory
before running `src/qcnn.py`. For example:

```python
import os
import tensorflow_quantum as tfq

data_dir = os.path.join(os.getcwd(), "tfi_chain")
tfq.datasets.tfi_chain(qubits, boundary_condition="closed", data_dir=data_dir)
```

The QCNN code expects files such as:

`tfi_chain/spin_systems/TFI_chain/closed/<qubits>/<gamma>/params.npy`
