import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import argparse
import os

def plot_dataset(input_file="tracks.csv", output_prefix="dataset_plot"):
    print(f"Loading data from {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        return

    # Check for required columns
    required_cols = ['t', 'x', 'y', 'z']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Dataset must contain {required_cols}")
        return

    # --- 1. Cell Count over Time ---
    print("Plotting Cell Count over Time...")
    cell_counts = df.groupby('t').size()
    
    plt.figure(figsize=(10, 6))
    plt.plot(cell_counts.index, cell_counts.values, 'b-', linewidth=2)
    plt.xlabel('Time (frames)')
    plt.ylabel('Number of Cells')
    plt.title('Cell Proliferation over Time')
    plt.grid(True)
    plt.savefig(f"{output_prefix}_cell_count.png")
    plt.close()

    # --- 2. Volume Distribution ---
    if 'volume' in df.columns:
        print("Plotting Volume Distribution...")
        plt.figure(figsize=(10, 6))
        sns.histplot(df['volume'], bins=50, kde=True, color='green')
        plt.xlabel('Cell Volume (pixels)')
        plt.ylabel('Count')
        plt.title('Distribution of Cell Volumes')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_prefix}_volume_dist.png")
        plt.close()
    
    # --- 3. 3D Structure at Key Frames ---
    print("Plotting 3D Structure at Key Frames...")
    times = sorted(df['t'].unique())
    if len(times) > 0:
        # Select 3 frames: Start, Middle, End
        frames_to_plot = [
            times[0],
            times[len(times)//2],
            times[-1]
        ]
        
        fig = plt.figure(figsize=(18, 6))
        
        for i, t in enumerate(frames_to_plot):
            ax = fig.add_subplot(1, 3, i+1, projection='3d')
            subset = df[df['t'] == t]
            
            # Scatter plot
            # If 'volume' exists, use it for size
            s = subset['volume'] / subset['volume'].mean() * 20 if 'volume' in df.columns else 20
            
            ax.scatter(subset['x'], subset['y'], subset['z'], s=s, alpha=0.6, edgecolors='w', linewidth=0.5)
            
            ax.set_title(f"Time = {t} (n={len(subset)})")
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            # Try to keep scale consistent if possible, or let it auto-scale
            ax.set_box_aspect([1,1,1]) 

        plt.suptitle("Embryo Structure at Different Stages")
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_3d_structure.png")
        plt.close()

    # --- 4. Trajectories ---
    print("Plotting Cell Trajectories...")
    # Select a few random cells that exist for a long time
    cell_ids = df['cell_id'].unique()
    
    # Filter cells that exist for at least 50% of the duration
    max_duration = df['t'].max() - df['t'].min()
    long_lived_cells = []
    for cid in cell_ids:
        c_data = df[df['cell_id'] == cid]
        if (c_data['t'].max() - c_data['t'].min()) > (0.5 * max_duration):
            long_lived_cells.append(cid)
            
    # Sample 50 cells
    if len(long_lived_cells) > 50:
        selected_cells = np.random.choice(long_lived_cells, 50, replace=False)
    else:
        selected_cells = long_lived_cells
        
    if len(selected_cells) > 0:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        for cid in selected_cells:
            track = df[df['cell_id'] == cid].sort_values('t')
            ax.plot(track['x'], track['y'], track['z'], alpha=0.5, linewidth=1)
            
        ax.set_title(f"Trajectories of {len(selected_cells)} Long-Lived Cells")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.savefig(f"{output_prefix}_trajectories.png")
        plt.close()
    else:
        print("Not enough long-lived cells found for trajectory plot.")

    print(f"Plots saved with prefix '{output_prefix}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="tracks.csv")
    parser.add_argument("--output_prefix", type=str, default="dataset_plot")
    args = parser.parse_args()
    
    plot_dataset(args.input_file, args.output_prefix)
