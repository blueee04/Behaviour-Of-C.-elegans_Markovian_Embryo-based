import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
import os
from tqdm import tqdm

def visualize_video(input_file, output_file="embryo_development.gif"):
    print(f"Loading tracks from {input_file}...")
    df = pd.read_csv(input_file)
    times = sorted(df['t'].unique())
    
    # Create temp directory for frames
    frames_dir = "frames"
    os.makedirs(frames_dir, exist_ok=True)
    
    images = []
    
    print("rendering frames...")
    # Use a fixed scale for the plot so the embryo doesn't "zoom in/out" wildly
    # We find the global bounds
    x_min, x_max = df['x'].min(), df['x'].max()
    y_min, y_max = df['y'].min(), df['y'].max()
    z_min, z_max = df['z'].min(), df['z'].max()
    
    # Add some padding
    pad = 20
    
    for t in tqdm(times):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        cells = df[df['t'] == t]
        
        # 1. Plot Cells
        ax.scatter(cells['x'], cells['y'], cells['z'], c='b', s=5, alpha=0.3, label='Cells')
        
        # 2. Compute and Plot Principal Axes
        if len(cells) >= 3:
            P = cells[['x', 'y', 'z']].values
            cm = np.mean(P, axis=0)
            P_centered = P - cm
            tensor = np.dot(P_centered.T, P_centered) / len(P)
            eigvals, eigvecs = np.linalg.eigh(tensor)
            
            # Sort
            idx = eigvals.argsort()[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]
            
            # Principal Axis 1 (Length) - Red
            v1 = eigvecs[:, 0] * np.sqrt(eigvals[0]) * 2 # Scale for visibility
            ax.quiver(cm[0], cm[1], cm[2], v1[0], v1[1], v1[2], color='r', linewidth=3, label='Length')
            
            # Principal Axis 2 (Width) - Orange
            v2 = eigvecs[:, 1] * np.sqrt(eigvals[1]) * 2
            ax.quiver(cm[0], cm[1], cm[2], v2[0], v2[1], v2[2], color='orange', linewidth=3, label='Width')
            
        # Set Consistent Bounds
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.set_zlim(z_min - pad, z_max + pad) # Z is usually small, so this might look flat
        
        ax.set_title(f"C. elegans Embryo Development - Frame {t}")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Save Frame
        filename = os.path.join(frames_dir, f"frame_{t:03d}.png")
        plt.savefig(filename)
        plt.close(fig)
        
        images.append(imageio.imread(filename))

    # Save GIF
    print(f"Saving video to {output_file}...")
    imageio.mimsave(output_file, images, duration=0.1) # 10 frames per second
    
    # Cleanup frames
    # for filename in glob.glob(os.path.join(frames_dir, "*.png")):
    #     os.remove(filename)
    # os.rmdir(frames_dir)
    print("Done.")

if __name__ == "__main__":
    visualize_video("tracks.csv")
