import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_symmetry(min_size, max_size, json_directory="."):
    """
    Load JSON results and visualize computation time and energy metrics.
    Includes computation of the time ratio between two methods.

    Args:
        json_directory (str): Path to the directory containing JSON files.
    """
    data = []

    print(f"Reading JSON files from '{json_directory}'...")

    # 1. Load and parse JSON files
    for size in range(min_size, max_size + 1):
        filename = f"rows{size}_cols{size}.json"
        filepath = os.path.join(json_directory, filename)

        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    content = json.load(f)
                    data.append({
                        'size': size,
                        'duration_sym': content.get('duration_sym'),
                        'duration_lqa': content.get('duration_lqa'),
                        'energy_sym': content.get('energy_sym'),
                        'energy_lqa': content.get('energy_lqa')
                    })
                print(f"  - Loaded: {filename}")
            except Exception as e:
                print(f"  - Warning: Failed to read {filename}: {e}")
        else:
            print(f"  - Warning: {filename} not found.")

    if not data:
        print("\nError: No valid data found. Plot generation aborted.")
        return

    # Data processing and ratio computation
    df = pd.DataFrame(data)
    df = df.sort_values(by='size').reset_index(drop=True)
    df['duration_ratio'] = df.apply(
        lambda row: row['duration_lqa'] / row['duration_sym'] if row['duration_sym'] > 0 else np.nan,
        axis=1
    )

    print("\nData processed. Generating plots...")
    sns.set_theme(style="whitegrid")

    # Plot: Duration vs. Size (with secondary axis for time ratio)
    fig, ax1 = plt.subplots(figsize=(14, 8))

    ax1.set_title('Calculation Duration & Time Ratio vs. Grid Size', fontsize=18, pad=20)
    ax1.set_xlabel('Grid Size (N x N)', fontsize=14)
    ax1.set_ylabel('Duration (seconds, log scale)', fontsize=14)

    sns.lineplot(data=df, x='size', y='duration_sym', marker='o', label='Duration (Symmetry)', ax=ax1,
                 color='royalblue')
    sns.lineplot(data=df, x='size', y='duration_lqa', marker='X', label='Duration (LQA)', ax=ax1,
                 color='darkorange')
    ax1.set_yscale('log')

    # Secondary y-axis: time ratio
    ax2 = ax1.twinx()
    ax2.set_ylabel('Time Ratio (LQA / Symmetry)', fontsize=14, color='green')

    sns.lineplot(data=df, x='size', y='duration_ratio', marker='.', linestyle='--',
                 label='Time Ratio (LQA/Sym)', ax=ax2, color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')
    ax1.get_legend().remove()

    duration_plot_filename = 'durations_vs_size_with_ratio.png'
    plt.savefig(duration_plot_filename, dpi=300)
    print(f"  - Saved: {duration_plot_filename}")

    # Plot: Energy vs. Size
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df, x='size', y='energy_sym', marker='o', label='Energy (Symmetry)')
    sns.lineplot(data=df, x='size', y='energy_lqa', marker='X', label='Energy (LQA)')
    plt.title('Final Energy vs. Grid Size', fontsize=16)
    plt.xlabel('Grid Size (N x N)', fontsize=12)
    plt.ylabel('Energy', fontsize=12)
    plt.legend()
    energy_plot_filename = 'energies_vs_size.png'
    plt.savefig(energy_plot_filename, dpi=300)
    print(f"  - Saved: {energy_plot_filename}")

    print("\nAll plots generated successfully.")
