.. Copyright 2022 <Huawei Technologies Co., Ltd>

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

API
===

API of the Python and C++ code of MindQuantum.
To have a Python frontend and C++ in the backend,
we make use of `Pybind11 <https://github.com/pybind/pybind11/>`_.
The C++ code represents quantum circuits with the help of
`Tweedledum <https://github.com/boschmitt/tweedledum>`_ networks, which can
also be turned into directed acyclic graphs
(`DAG <https://en.wikipedia.org/wiki/Directed_acyclic_graph>`_),
to make some operations more efficient.

.. toctree::
   :maxdepth: 1

   api/python
   api/cxx
