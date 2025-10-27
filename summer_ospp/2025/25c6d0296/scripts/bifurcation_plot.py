import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_json(file_path="bifurcation_tree_data.json"):
    """
    Loads bifurcation history data from a JSON file and converts path data back into NumPy arrays.
    """
    print(f"--- Loading data from {file_path}... ---")
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Data file not found at '{file_path}'. Please run the main program first to generate the data.")
        return None

    # Convert path and vector data from lists back to NumPy arrays
    for record in data:
        if 'path_data' in record:
            record['path_data'] = np.array(record['path_data'])
        if record.get('bifurcation_info'):
            bif_info = record['bifurcation_info']
            if bif_info and bif_info.get('z') is not None:
                bif_info['z'] = np.array(bif_info['z'])
            if bif_info and bif_info.get('v_list') is not None:
                bif_info['v_list'] = [np.array(v) for v in bif_info['v_list']]
    print("--- Data loading and conversion complete. ---")
    return data

def plot_bifurcation_diagram(records, save_path, var_index=0):
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.viridis(np.linspace(0, 1, len(records) + 1))
    bif_points = [r['bifurcation_info']['z'] for r in records if r.get('bifurcation_info')]
    for i, record in enumerate(records):
        path_data = record['path_data']
        if path_data.shape[0] > 1:
            ax.plot(path_data[:, -1], path_data[:, var_index], '-', color=colors[i], lw=1.5, alpha=0.8)
    if bif_points:
        p_bifs = [z[-1] for z in bif_points]
        x_bifs = [z[var_index] for z in bif_points]
        ax.plot(p_bifs, x_bifs, 'rs', markersize=6, label='Bifurcation Points', zorder=10)
        ax.legend()
    ax.set_title(f'Bifurcation Diagram for $x_{{{var_index}}}$')
    ax.set_xlabel('Homotopy Parameter $p$')
    ax.set_ylabel(f'Value of $x_{{{var_index}}}$')
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Bifurcation diagram saved to: {save_path}")

def visualize_pca_trajectory(tree_data, save_path="pca_trajectory.png"):
    """
    Performs PCA dimensionality reduction on the entire bifurcation tree path and colors the trajectory based on cumulative arc length.
    """
    if not tree_data:
        print("Data is empty. Skipping visualization.")
        return

    print("\n--- Preparing for PCA and cumulative arc length calculation... ---")

    # 1. Reconstruct the parent-child relationships of the tree and calculate the intrinsic arc length of each branch.
    paths_by_id = {record['branch_id']: record for record in tree_data}
    children_map = {parent_id: [] for parent_id in paths_by_id}
    branch_self_lengths = {}

    for branch_id, record in paths_by_id.items():
        if record['parent_id'] in children_map:
            children_map[record['parent_id']].append(branch_id)

        path = record['path_data']
        # Calculate the arc length of this branch (sum of Euclidean distances between consecutive points).
        distances = np.linalg.norm(np.diff(path, axis=0), axis=1)
        branch_self_lengths[branch_id] = np.sum(distances)

    # 2. Recursively calculate the cumulative arc length for each point.
    all_x_vectors = []
    all_cumulative_lengths = []
    bifurcation_points_z = []

    def traverse_and_calculate(branch_id, parent_cumulative_length):
        record = paths_by_id[branch_id]
        path = record['path_data']

        # Calculate the local arc length for each point on the current path.
        local_lengths = np.zeros(len(path))
        distances = np.linalg.norm(np.diff(path, axis=0), axis=1)
        local_lengths[1:] = np.cumsum(distances)

        # Cumulative arc length = total length of parent path + local arc length of current path.
        cumulative_lengths = parent_cumulative_length + local_lengths

        all_x_vectors.extend(path[:, :-1])  # Extract all x vectors.
        all_cumulative_lengths.extend(cumulative_lengths)

        # If the path terminates at a bifurcation point, record it.
        if record['status'] == 'BIFURCATION' and record['bifurcation_info']:
            bifurcation_points_z.append(record['bifurcation_info']['z'])

        # Recursively explore child paths.
        for child_id in children_map.get(branch_id, []):
            # The starting point of a child path has a cumulative arc length equal to the total length of its parent path.
            child_start_length = parent_cumulative_length + branch_self_lengths[branch_id]
            traverse_and_calculate(child_id, child_start_length)

    # Start traversal from the root node (ID=0) with an initial arc length of 0.
    traverse_and_calculate(0, 0.0)

    all_x_vectors = np.array(all_x_vectors)
    print(f"--- Collected a total of {len(all_x_vectors)} data points for PCA analysis... ---")

    # 3. Perform PCA.
    # Standardize the data first, which is a best practice for PCA.
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(all_x_vectors)

    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x_scaled)

    print(f"--- PCA complete. Explained variance ratio by Principal Component 1: {pca.explained_variance_ratio_[0]:.2%} ---")
    print(f"--- Explained variance ratio by Principal Component 2: {pca.explained_variance_ratio_[1]:.2%} ---")

    # 4. Plotting.
    print("\n--- Commencing PCA trajectory plot generation... ---")
    fig, ax = plt.subplots(figsize=(16, 9))

    # Use a scatter plot for visualization, with color varying according to cumulative arc length.
    sc = ax.scatter(x_pca[:, 0], x_pca[:, 1],
                    c=all_cumulative_lengths,
                    cmap='jet',  # Colormap 'jet': Blue -> Cyan -> Green -> Yellow -> Red.
                    s=5, alpha=0.7)

    # Add a color bar.
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Cumulative Arc Length')

    # Mark the bifurcation points.
    if bifurcation_points_z:
        bif_x_vectors = [z[:-1] for z in bifurcation_points_z]
        # The coordinates of the bifurcation points must undergo the same standardization and PCA transformation.
        bif_x_scaled = scaler.transform(bif_x_vectors)
        bif_x_pca = pca.transform(bif_x_scaled)
        ax.scatter(bif_x_pca[:, 0], bif_x_pca[:, 1], c='black', marker='s', s=3, label='Bifurcation Points')

    ax.set_title('PCA Projection of Solution Paths Trajectory')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.grid(True, linestyle='--', alpha=0.6)

    try:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"--- PCA trajectory plot successfully saved to: {save_path} ---")
    except Exception as e:
        print(f"--- Failed to save the image: {e} ---")
    plt.close(fig)


def visualize_pca_p_colored(tree_data, save_path="pca_p_colored.png"):
    """
    Performs PCA dimensionality reduction on the entire bifurcation tree path and colors the trajectory based on the homotopy parameter 'p'.
    """
    if not tree_data:
        print("Data is empty. Skipping visualization.")
        return

    print("\n--- Preparing for PCA and color mapping based on p-value... ---")

    # 1. Collect all x vectors and their corresponding p-values from the tree data structure.
    all_x_vectors = []
    all_p_values = []
    bifurcation_points_z = []

    for record in tree_data:
        path_data = record['path_data']
        if path_data.shape[0] > 1:
            all_x_vectors.extend(path_data[:, :-1])  # Extract all x vectors.
            all_p_values.extend(path_data[:, -1])  # Extract all p-values.

        if record['status'] == 'BIFURCATION' and record['bifurcation_info']:
            bif_info = record['bifurcation_info']
            if bif_info and bif_info.get('z') is not None:
                bifurcation_points_z.append(bif_info['z'])

    all_x_vectors = np.array(all_x_vectors)
    print(f"--- Collected a total of {len(all_x_vectors)} data points for PCA analysis... ---")

    # 2. Perform PCA (identical to the previous function).
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(all_x_vectors)
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x_scaled)

    print(f"--- PCA complete. ---")

    # 3. Plotting.
    print("\n--- Commencing generation of the p-colored PCA trajectory plot... ---")
    fig, ax = plt.subplots(figsize=(16, 9))

    # --- Core modification: Use all_p_values as the color parameter 'c'. ---
    sc = ax.scatter(x_pca[:, 0], x_pca[:, 1],
                    c=all_p_values,
                    cmap='jet',  # Colormap 'jet': Blue (p=0) -> Green -> Yellow -> Red (p=1).
                    s=5, alpha=0.7)

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Homotopy Parameter (p)')

    # Mark the bifurcation points.
    if bifurcation_points_z:
        bif_x_vectors = [z[:-1] for z in bifurcation_points_z]
        bif_x_scaled = scaler.transform(bif_x_vectors)
        bif_x_pca = pca.transform(bif_x_scaled)
        ax.scatter(bif_x_pca[:, 0], bif_x_pca[:, 1],
                   facecolors='none', edgecolors='black',  # Use hollow black squares for distinction.
                   marker='s', s=5,
                   label='Bifurcation Points')

    ax.set_title('PCA Projection of Solution Paths (Colored by Parameter p)')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.grid(True, linestyle='--', alpha=0.6)

    try:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"--- The p-colored PCA trajectory plot has been successfully saved to: {save_path} ---")
    except Exception as e:
        print(f"--- Failed to save the image: {e} ---")
    plt.close(fig)


def calculate_energy_history(tree_data, hubo_problem):
    """
    Iterates through the loaded tree data to compute the energy history for each branch.
    This function modifies the input 'tree_data' in place.
    """
    print("\n--- Commencing energy calculation for all path points... ---")
    for record in tree_data:
        energy_history = []
        # 'path_data' contains multiple vectors of the form [x..., p].
        for step_data in record['path_data']:
            x = step_data[:-1]  # Extract the solution vector x.
            p = step_data[-1]   # Extract the parameter p.

            # Calculate the energy using the provided method.
            E = hubo_problem.energy(x)

            energy_history.append([p, E])

        # Store the computed energy history back into the record.
        record['energy_history'] = energy_history
    print("--- Energy calculation complete. ---")


def plot_energy_landscape(tree_data, save_path="energy_landscape.png"):
    """
    Plots an energy landscape (Energy vs. p) based on the branch history data.
    """
    print("\n--- Commencing generation of the energy landscape plot... ---")
    fig, ax = plt.subplots(figsize=(16, 9))
    colors = plt.cm.turbo(np.linspace(0, 1, len(tree_data)))  # Use the 'turbo' colormap.

    for i, record in enumerate(tree_data):
        # Note that we use 'energy_history' here.
        history = np.array(record.get('energy_history', []))
        if history.shape[0] > 1:
            p_values = history[:, 0]
            energy_values = history[:, 1]
            ax.plot(p_values, energy_values, '-', color=colors[i], alpha=0.8,
                    label=f"Branch {record['branch_id']}" if len(tree_data) < 15 else None)

    ax.set_title('Energy Landscape (H(x) vs. p)')
    ax.set_xlabel('Homotopy Parameter (p)')
    ax.set_ylabel('HUBO Energy H(x)')
    ax.grid(True, linestyle='--', alpha=0.6)
    if len(tree_data) < 15:
        ax.legend()

    try:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"--- Energy landscape plot successfully saved to: {save_path} ---")
    except Exception as e:
        print(f"--- Failed to save the image: {e} ---")
    plt.close(fig)
