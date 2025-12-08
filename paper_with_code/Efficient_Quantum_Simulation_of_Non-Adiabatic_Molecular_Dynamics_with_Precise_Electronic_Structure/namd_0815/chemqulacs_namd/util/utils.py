# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pubchempy


def get_geometry_from_pubchem(name):
    """
    Get a geometry file from PubChem using pubchempy

    Args:
      name: name of molecule

    Returns:
      numpy array: two-dimensional arrays of coordinates

    Examples:

      >>> geom_water = get_geometry_from_pubchem('water')
      >>> mol = gto.M(atom=geom_water,basis='sto-3g')

    """
    pubchempy_molecule = pubchempy.get_compounds(name, "name", record_type="3d")
    pubchempy_geometry = pubchempy_molecule[0].to_dict(properties=["atoms"])["atoms"]
    geometry = [
        (atom["element"], (atom["x"], atom["y"], atom.get("z", 0)))
        for atom in pubchempy_geometry
    ]
    return geometry


def almost_equal(x, y, threshold=0.00000001):
    """
    Returns if two arrays are element-wise equal within a ``threshold``.
    """
    return abs(x - y) < threshold
