import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

def analyze_embryo(input_file, output_prefix="embryo_analysis"):
    print(f"Loading tracks from {input_file}...")
    df = pd.read_csv(input_file)
    
    times = sorted(df['t'].unique())
    eigenvalues_over_time = []
    
    # Store principal axes for visualization (maybe for a few frames)
    # But mainly we want the eigenvalues
    
    print("Performing Shape Tensor Analysis...")
    for t in tqdm(times):
        # Get cells at this time point
        cells = df[df['t'] == t]
        
        if len(cells) < 3:
            # Not enough points for 3D PCA
            eigenvalues_over_time.append({
                't': t, 'l1': 0, 'l2': 0, 'l3': 0, 
                'rate_of_change': 0,
                'num_cells': len(cells)
            })
            continue
            
        # Coordinates
        P = cells[['x', 'y', 'z']].values
        
        # Center of Mass
        cm = np.mean(P, axis=0)
        
        # Centered clouds
        P_centered = P - cm
        
        # Gyration Tensor / Inertia Tensor (Covariance matrix of positions)
        # S = sum((r_i - r_cm)(r_i - r_cm)^T)
        # In code: P_centered.T @ P_centered
        tensor = np.dot(P_centered.T, P_centered) / len(P)
        
        # Eigen decomposition
        # eigh is for symmetric matrices (Hermitian)
        eigvals, eigvecs = np.linalg.eigh(tensor)
        
        # Sort eigenvalues descending (lambda1 >= lambda2 >= lambda3)
        # np.linalg.eigh returns them in ascending order usually, so reverse
        idx = eigvals.argsort()[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        eigenvalues_over_time.append({
            't': t,
            'l1': eigvals[0],
            'l2': eigvals[1],
            'l3': eigvals[2],
            'num_cells': len(cells)
        })

    # Create DataFrame for results
    results = pd.DataFrame(eigenvalues_over_time)
    
    # Calculate Dynamics (Rate of change of shape)
    # We can approximate this as the norm of the derivative of the eigenvalue vector
    # Lambda = [l1, l2, l3]
    # Velocity = |dLambda/dt|
    l_vec = results[['l1', 'l2', 'l3']].values
    # Simple difference
    delta_l = np.diff(l_vec, axis=0)
    # Prepend 0 for the first frame
    velocity = np.linalg.norm(delta_l, axis=1)
    velocity = np.insert(velocity, 0, 0)
    
    results['shape_velocity'] = velocity

    # --- Plotting ---
    print("Generating plots...")
    
    # Plot 1: Standard deviation lengths (sqrt of eigenvalues) = roughly dimensions of the embryo
    plt.figure(figsize=(10, 6))
    plt.plot(results['t'], np.sqrt(results['l1']), label='Length (PC1)')
    plt.plot(results['t'], np.sqrt(results['l2']), label='Width (PC2)')
    plt.plot(results['t'], np.sqrt(results['l3']), label='Depth (PC3)')
    plt.xlabel('Time (frames)')
    plt.ylabel('Principal Axis Length (pixels)')
    plt.title('Embryo Shape Evolution (Principal Axes)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_prefix}_dimensions.png")
    
    # Plot 2: Shape Space Trajectory (l1 vs l2)
    plt.figure(figsize=(8, 8))
    plt.plot(np.sqrt(results['l1']), np.sqrt(results['l2']), '-o', markersize=2, alpha=0.5)
    plt.xlabel('Principal Axis 1 (Length)')
    plt.ylabel('Principal Axis 2 (Width)')
    plt.title('Trajectory in Shape Eigenspace')
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(f"{output_prefix}_eigenspace_trajectory.png")
    
    # Plot 3: Shape Velocity (Developmental Activity)
    plt.figure(figsize=(10, 6))
    plt.plot(results['t'], results['shape_velocity'], color='red')
    plt.xlabel('Time')
    plt.ylabel('Rate of Change of Shape (Velocity)')
    plt.title('Developmental Activity (Shape Dynamics)')
    plt.grid(True)
    plt.savefig(f"{output_prefix}_shape_velocity.png")
    
    print(f"Analysis complete. Plots saved to current directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="tracks.csv")
    args = parser.parse_args()
    
    analyze_embryo(args.input_file)
