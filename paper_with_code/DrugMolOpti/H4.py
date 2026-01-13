from geo_fun import DMETVQEGeometryOptimizer
import numpy as np

def geometry_to_xyz(para):
    """Given geometry para = [bond_1, bond_2], construct the XYZ string for H4."""
    theta_1 = 0.0
    theta_2 = 0.0

    bond_1, bond_2 = para

    x_H3 = bond_2 * np.cos(theta_1)
    y_H3 = bond_2 * np.sin(theta_1)
    x_H4 = x_H3 + bond_1 * np.cos(theta_2)
    y_H4 = y_H3 + bond_1 * np.sin(theta_2)

    mo_all = f"""
    H          {-1 * bond_1}   0.0    0.0
    H          0.0    0.0    0.0
    H          {x_H3}    {y_H3}    0.0 
    H          {x_H4}    {y_H4}    0.0 
    """
    return mo_all


optimizer = DMETVQEGeometryOptimizer(
    para_file="./para_var.npy",
    geometry_fn=geometry_to_xyz,  # inject custom geometry
)
optimizer.init_para = [1.0, 1.0]

# Geometry bounds

optimizer.special_bounds = [
    (0.5, 2.0),  # bond_1
    (0.5, 3.0),  # bond_2
]

# DMET fragment settings: 4 atoms, each in its own fragment
optimizer.fragment_atoms = [1] * 4

optimizer.solve_chemical_potential(mu0=0.0, tol=1e-5)
