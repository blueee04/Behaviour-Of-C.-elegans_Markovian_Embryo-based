import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2
import os
import glob
from skimage.io import imread
from skimage.exposure import rescale_intensity
from tqdm import tqdm
import argparse

def create_video(results_csv, data_dir, output_file="embryo_development.mp4", fps=10):
    print(f"Loading results from {results_csv}...")
    try:
        df = pd.read_csv(results_csv)
    except FileNotFoundError:
        print("Error: Results file not found.")
        return

    print(f"Searching for TIFFs in {data_dir}...")
    # Assume filenames are t000.tif, t001.tif, etc.
    
    # Setup Video Writer
    # We'll determine frame size from the first generated plot
    writer = None
    
    # Filter times to those present in both CSV and directory (conceptually)
    # We iterate through the CSV times
    times = df['t'].values
    
    # Pre-calculate global min/max for plots to keep axes valid
    l1_micro_min, l1_micro_max = df['l1_micro'].min(), df['l1_micro'].max()
    l1_sqrt = np.sqrt(df['l1'])
    l2_sqrt = np.sqrt(df['l2'])
    
    # Behavioral State Cluster Colors (Use the 'state' column)
    # Map states to colors
    unique_states = sorted(df['state'].unique())
    state_colors = plt.cm.get_cmap('tab10', len(unique_states))

    print(f"Generating video with {len(times)} frames...")
    
    for i, t in enumerate(tqdm(times)):
        # 1. Load Raw Image
        filename = f"t{int(t):03d}.tif"
        filepath = os.path.join(data_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"Warning: File {filename} not found. Skipping.")
            continue
            
        try:
            img = imread(filepath)
            mip = np.max(img, axis=0)
            p2, p98 = np.percentile(mip, (2, 98))
            mip = rescale_intensity(mip, in_range=(p2, p98))
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue

        # 2. Setup Figure
        fig = plt.figure(figsize=(16, 9))
        gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
        
        # --- Left Panel: Raw Data (Full Height) ---
        ax_raw = fig.add_subplot(gs[:, 0])
        ax_raw.imshow(mip, cmap='gray')
        ax_raw.set_title(f"Raw Embryo Data (Frame {int(t)})", fontsize=16)
        ax_raw.axis('off')
        
        # --- Top Right: Micro-dynamics Trace ---
        ax_micro = fig.add_subplot(gs[0, 1])
        # Plot full trace as faint background
        ax_micro.plot(df['t'], df['l1_micro'], color='gray', alpha=0.3)
        # Plot trace up to current time
        current_trace = df[df['t'] <= t]
        ax_micro.plot(current_trace['t'], current_trace['l1_micro'], color='blue', linewidth=2)
        # Highlight current point
        curr_val = df[df['t'] == t]['l1_micro'].values[0]
        ax_micro.plot(t, curr_val, 'ro', markersize=8)
        
        ax_micro.set_xlim(df['t'].min(), df['t'].max())
        ax_micro.set_ylim(l1_micro_min, l1_micro_max)
        ax_micro.set_xlabel("Time")
        ax_micro.set_ylabel("Fluctuation (pixels)")
        ax_micro.set_title("Micro-dynamics (Length Fluctuations)", fontsize=14)
        ax_micro.grid(True, alpha=0.3)
        
        # --- Bottom Right: Behavioral State Space ---
        ax_state = fig.add_subplot(gs[1, 1])
        
        # Scatter all points colored by state
        scatter = ax_state.scatter(
            l1_sqrt, l2_sqrt, 
            c=df['state'], cmap='tab10', 
            s=30, alpha=0.4, edgecolors='none'
        )
        
        # Current position
        curr_l1 = l1_sqrt.iloc[i]
        curr_l2 = l2_sqrt.iloc[i]
        ax_state.plot(curr_l1, curr_l2, 'ko', markersize=10, markerfacecolor='white', markeredgewidth=2)
        
        # Trajectory trail (last 20 frames)
        if i > 0:
            trail_start = max(0, i-20)
            ax_state.plot(l1_sqrt.iloc[trail_start:i+1], l2_sqrt.iloc[trail_start:i+1], 'k-', linewidth=1, alpha=0.6)

        ax_state.set_xlabel("Length (PC1)")
        ax_state.set_ylabel("Width (PC2)")
        ax_state.set_title(f"Behavioral State Space (Current State: {int(df.iloc[i]['state'])})", fontsize=14)
        ax_state.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 3. Convert to Image array for Video Writing
        fig.canvas.draw()
        
        # Extract RGBA buffer (modern Matplotlib)
        # buffer_rgba() returns a memoryview
        buf = np.array(fig.canvas.buffer_rgba())
        
        # Convert RGBA to BGR for OpenCV
        frame_bgr = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
        
        plt.close(fig)
        
        # 4. Initialize Writer if needed
        if writer is None:
            h, w = frame_bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or 'avc1' or 'XVID'
            writer = cv2.VideoWriter(output_file, fourcc, fps, (w, h))
            
        writer.write(frame_bgr)

    if writer:
        writer.release()
        print(f"Video saved to {output_file}")
    else:
        print("No video created.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="embryo_analysis_results.csv")
    parser.add_argument("--data_dir", type=str, default=r"d:\Github\Behaviour-Of-C.-elegans_Markovian_Embryo-based\Data\Train_Data\Fluo-N3DH-CE\01")
    parser.add_argument("--output", type=str, default="embryo_development.mp4")
    parser.add_argument("--fps", type=int, default=15)
    
    args = parser.parse_args()
    
    create_video(args.results, args.data_dir, args.output, args.fps)
