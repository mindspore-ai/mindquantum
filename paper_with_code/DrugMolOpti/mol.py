import numpy as np

def geometry_to_xyz(para):
    theta_1 = para[0]
    theta_2 = para[1]
    bond_1 = para[2]
    bond_2 = para[3]

    x_H2 = -1 * bond_1 * np.cos(theta_1) * np.sin(theta_2)
    y_H2 = bond_2 + bond_1 * np.cos(theta_1) * np.cos(theta_2)
    z_H2 =  bond_1 * np.sin(theta_1)

    x_H1 = -1 * x_H2
    y_H1 = -1 * bond_1 * np.cos(theta_1) * np.cos(theta_2)
    z_H1 = z_H2

    mo_all = f"""
    H          {x_H1}    {y_H1}    {z_H1} 
    O          0.0    0.0    0.0
    O          0.0    {bond_2}    0.0 
    H          {x_H2}    {y_H2}    {z_H2} 
    """
    return mo_all


def geometry_to_xyz_2(para):

    C_ref_abs = np.array([-0.7521, -0.7524, -0.0119])

    O1_rel_standard = np.array([-1.8877 - C_ref_abs[0], 
                                  0.0894 - C_ref_abs[1], 
                                  0.0097 - C_ref_abs[2]])


    H3_rel_standard = np.array([-1.8945 - C_ref_abs[0], 
                                  0.5504 - C_ref_abs[1], 
                                  0.8660 - C_ref_abs[2]])



    angle_z = para[0] # 绕 Z 轴的旋转角
    angle_y = para[1] # 绕 Y 轴的旋转角

    # 绕 Z 轴的旋转矩阵
    R_z = np.array([
        [np.cos(angle_z), -np.sin(angle_z), 0],
        [np.sin(angle_z),  np.cos(angle_z), 0],
        [0, 0, 1]
    ])

    R_y = np.array([
        [np.cos(angle_y), 0, np.sin(angle_y)],
        [0, 1, 0],
        [-np.sin(angle_y), 0, np.cos(angle_y)]
    ])

    rotation_matrix = np.dot(R_y, R_z)

    rotated_O1_rel = np.dot(rotation_matrix, O1_rel_standard)
    rotated_H3_rel = np.dot(rotation_matrix, H3_rel_standard)

    x_O1, y_O1, z_O1 = rotated_O1_rel + C_ref_abs
    x_H3, y_H3, z_H3 = rotated_H3_rel + C_ref_abs

    mo_all = f"""
    O          {x_O1}  {y_O1:}    {z_O1}
    O          1.5950   -0.7195    0.0078
    O          0.5455    1.3018   -0.0031
    C          {C_ref_abs[0]}   {C_ref_abs[1]}   {C_ref_abs[2]}
    C          0.4994    0.0808   -0.0026
    H          -0.7773   -1.4045    0.8654
    H          -0.7835   -1.3486   -0.9275
    H          {x_H3}    {y_H3}    {z_H3}
    H          2.4274   -0.2005    0.0146
    """
    return mo_all
