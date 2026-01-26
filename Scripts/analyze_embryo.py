import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
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

    # --- Micro-dynamics / Micro-state Analysis ---
    print("Performing Micro-dynamics Analysis...")
    try:
        from scipy.signal import savgol_filter
        
        # We analyze fluctuations in the Principal Axis lengths (sqrt(eigenvalues))
        # l1_len = sqrt(l1), etc.
        l1_len = np.sqrt(results['l1'])
        l2_len = np.sqrt(results['l2'])
        l3_len = np.sqrt(results['l3'])
        
        # 1. Calculate Macro-state (Smooth Trend)
        # Window size must be odd. Adjust based on frame rate. 
        # If we have 195 frames, a window of ~21 frames seems reasonable to capture "slow" growth.
        window_size = 21 
        poly_order = 3
        
        l1_trend = savgol_filter(l1_len, window_size, poly_order)
        l2_trend = savgol_filter(l2_len, window_size, poly_order)
        l3_trend = savgol_filter(l3_len, window_size, poly_order)
        
        # 2. Calculate Micro-state (Residuals / Fluctuations)
        l1_micro = l1_len - l1_trend
        l2_micro = l2_len - l2_trend
        l3_micro = l3_len - l3_trend

        # Add to results DataFrame
        results['l1_micro'] = l1_micro
        results['l2_micro'] = l2_micro
        results['l3_micro'] = l3_micro
        
        # Plot 4: Micro-state Fluctuations
        plt.figure(figsize=(12, 6))
        plt.plot(results['t'], l1_micro, label='Length Fluctuations (PC1)', alpha=0.8)
        plt.plot(results['t'], l2_micro, label='Width Fluctuations (PC2)', alpha=0.8)
        plt.plot(results['t'], l3_micro, label='Depth Fluctuations (PC3)', alpha=0.8)
        plt.xlabel('Time (frames)')
        plt.ylabel('Deviation from Trend (pixels)')
        plt.title('Micro-dynamics: High-Frequency Shape Fluctuations')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_prefix}_micro_fluctuations.png")
        
        # Plot 5: Fluctuation Intensity (Variance of micro-states over sliding window)
        # This helps see *when* the embryo is most active/twitching
        window = 5
        activity = pd.Series(l1_micro).rolling(window=window).std().fillna(0) + \
                   pd.Series(l2_micro).rolling(window=window).std().fillna(0)
        
        plt.figure(figsize=(10, 6))
        plt.plot(results['t'], activity, color='purple')
        plt.xlabel('Time (frames)')
        plt.ylabel('Fluctuation Intensity (std dev)')
        plt.title('Micro-state Activity Levels (Twitching Analysis)')
        plt.fill_between(results['t'], activity, color='purple', alpha=0.2)
        plt.grid(True)
        plt.savefig(f"{output_prefix}_activity_intensity.png")
        
        print("Micro-dynamics analysis complete.")
        
    except ImportError:
        print("Error: Scipy not installed. Skipping micro-dynamics analysis.")
    except Exception as e:
        print(f"Error in micro-dynamics analysis: {e}")

    # --- Behavioral State Analysis (Clustering) ---
    print("Performing Behavioral State Analysis (Clustering)...")
    try:
        from sklearn.cluster import KMeans
        import seaborn as sns
        
        # We cluster based on the Shape Space coordinates: sqrt(l1) (Length) and sqrt(l2) (Width)
        # These are the "Principal Components" of the shape.
        X = np.column_stack((np.sqrt(results['l1']), np.sqrt(results['l2'])))
        
        # K=4 for Bean, Comma, 1.5-fold, 2-fold (Approximate stages)
        k = 4
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        results['cluster'] = labels
        
        # Sort clusters by time to make colors consistent (0=Early, K=Late)
        # We find the mean time for each cluster and re-map labels
        cluster_times = results.groupby('cluster')['t'].mean().sort_values()
        mapping = {old: new for new, old in enumerate(cluster_times.index)}
        results['state'] = results['cluster'].map(mapping)
        
        # Plot 6: Embryo Behavioral States (The "User Style" Plot)
        plt.figure(figsize=(10, 8))
        
        # Create a custom color palette similar to the user's reference (Earth tones / Distinct)
        palette = sns.color_palette("deep", k)
        
        # Scatter plot with colored clusters
        sns.scatterplot(
            x=np.sqrt(results['l1']), 
            y=np.sqrt(results['l2']), 
            hue=results['state'], 
            palette=palette,
            s=50,
            edgecolor='k',
            alpha=0.8,
            legend='full'
        )
        
        # Draw Trajectory Line connecting points
        plt.plot(np.sqrt(results['l1']), np.sqrt(results['l2']), 'k-', alpha=0.2, linewidth=1)
        
        # Annotate States (Centroids)
        state_names = {
            0: "1. Early/Bean",
            1: "2. Comma",
            2: "3. 1.5-Fold", 
            3: "4. 2-Fold+"
        }
        
        # Calculate centroids of the *new* mapped states
        for state_id in range(k):
            state_data = results[results['state'] == state_id]
            if len(state_data) == 0: continue
            
            centroid_x = np.mean(np.sqrt(state_data['l1']))
            centroid_y = np.mean(np.sqrt(state_data['l2']))
            
            plt.text(
                centroid_x, centroid_y, 
                state_names.get(state_id, f"State {state_id}"), 
                fontsize=12, 
                fontweight='bold', 
                color='black',
                ha='center', 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3)
            )

        plt.xlabel('Length (Principal Axis 1)')
        plt.ylabel('Width (Principal Axis 2)')
        plt.title('Embryo "Behavioral States" (Morphological Stages)')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(f"{output_prefix}_behavioral_states.png")
        print(f"State map saved to {output_prefix}_behavioral_states.png")

    except ImportError:
        print("Error: sklearn or seaborn not installed.")
    except Exception as e:
        print(f"Error in Clustering analysis: {e}")

    # --- Save Full Results ---
    output_csv = f"{output_prefix}_results.csv"
    print(f"Saving full analysis results to {output_csv}...")
    results.to_csv(output_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="tracks.csv")
    args = parser.parse_args()
    
    analyze_embryo(args.input_file)
