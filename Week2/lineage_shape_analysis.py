"""
Lineage-Specific Shape Analysis for C. elegans Embryo

This module performs shape tensor analysis on individual lineage sub-clouds
(e.g., all AB descendants, all P1 descendants, etc.) to understand
lineage-specific morphological dynamics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
from sklearn.cluster import KMeans
import os

def analyze_lineage_cloud(df_lineage, lineage_name, output_dir):
    """
    Perform shape tensor analysis on a lineage sub-cloud.
    
    Args:
        df_lineage: DataFrame with columns [t, x, y, z] for all cells in this lineage
        lineage_name: string (e.g., "AB", "P1")
        output_dir: directory to save plots
        
    Returns:
        DataFrame with time-series analysis results
    """
    print(f"\nAnalyzing {lineage_name} lineage...")
    
    times = sorted(df_lineage['t'].unique())
    if len(times) < 3:
        print(f"  Insufficient data points ({len(times)} frames)")
        return None
        
    results = []
    
    for t in times:
        cells = df_lineage[df_lineage['t'] == t]
        
        if len(cells) < 2:
            # Need at least 2 points for meaningful analysis
            results.append({
                't': t,
                'l1': 0, 'l2': 0, 'l3': 0,
                'num_cells': len(cells)
            })
            continue
            
        # Get positions
        P = cells[['x', 'y', 'z']].values
        
        # Center of mass
        cm = np.mean(P, axis=0)
        P_centered = P - cm
        
        # Shape tensor (gyration tensor)
        tensor = np.dot(P_centered.T, P_centered) / len(P)
        
        # Eigenvalues
        eigvals, eigvecs = np.linalg.eigh(tensor)
        eigvals = eigvals[::-1]  # Sort descending
        
        results.append({
            't': t,
            'l1': eigvals[0] if len(eigvals) > 0 else 0,
            'l2': eigvals[1] if len(eigvals) > 1 else 0,
            'l3': eigvals[2] if len(eigvals) > 2 else 0,
            'num_cells': len(cells),
            'cm_x': cm[0],
            'cm_y': cm[1],
            'cm_z': cm[2]
        })
        
    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    if len(df_results) < 10:
        print(f"  Insufficient time points ({len(df_results)} frames)")
        return df_results
        
    # Calculate shape velocity
    l_vec = df_results[['l1', 'l2', 'l3']].values
    delta_l = np.diff(l_vec, axis=0)
    velocity = np.linalg.norm(delta_l, axis=1)
    velocity = np.insert(velocity, 0, 0)
    df_results['shape_velocity'] = velocity
    
    # Calculate micro-dynamics (if enough points)
    if len(df_results) >= 21:
        try:
            l1_len = np.sqrt(df_results['l1'])
            l2_len = np.sqrt(df_results['l2'])
            l3_len = np.sqrt(df_results['l3'])
            
            window_size = min(21, len(df_results) // 2 * 2 - 1)  # Ensure odd
            if window_size >= 3:
                l1_trend = savgol_filter(l1_len, window_size, 3)
                l2_trend = savgol_filter(l2_len, window_size, 3)
                l3_trend = savgol_filter(l3_len, window_size, 3)
                
                df_results['l1_micro'] = l1_len - l1_trend
                df_results['l2_micro'] = l2_len - l2_trend
                df_results['l3_micro'] = l3_len - l3_trend
        except:
            pass
            
    # Clustering (if enough points)
    if len(df_results) >= 10:
        try:
            X = np.column_stack((np.sqrt(df_results['l1']), np.sqrt(df_results['l2'])))
            k = min(3, len(df_results) // 5)  # Adaptive k
            if k >= 2:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
                df_results['cluster'] = kmeans.fit_predict(X)
        except:
            pass
            
    # Generate plots
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Principal axes over time
    plt.figure(figsize=(10, 6))
    plt.plot(df_results['t'], np.sqrt(df_results['l1']), label='Length (L1)', linewidth=2)
    plt.plot(df_results['t'], np.sqrt(df_results['l2']), label='Width (L2)', linewidth=2)
    plt.plot(df_results['t'], np.sqrt(df_results['l3']), label='Depth (L3)', linewidth=2)
    plt.xlabel('Time (frames)', fontsize=12)
    plt.ylabel('Principal Axis Length (pixels)', fontsize=12)
    plt.title(f'{lineage_name} Lineage: Shape Evolution', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{lineage_name}_dimensions.png'), dpi=150)
    plt.close()
    
    # Plot 2: Cell count over time
    plt.figure(figsize=(10, 6))
    plt.plot(df_results['t'], df_results['num_cells'], 'o-', linewidth=2, markersize=4)
    plt.xlabel('Time (frames)', fontsize=12)
    plt.ylabel('Number of Cells', fontsize=12)
    plt.title(f'{lineage_name} Lineage: Cell Proliferation', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{lineage_name}_cell_count.png'), dpi=150)
    plt.close()
    
    # Plot 3: Shape space trajectory
    plt.figure(figsize=(8, 8))
    plt.plot(np.sqrt(df_results['l1']), np.sqrt(df_results['l2']), '-o', 
             markersize=3, alpha=0.6, linewidth=1)
    plt.xlabel('Length (L1)', fontsize=12)
    plt.ylabel('Width (L2)', fontsize=12)
    plt.title(f'{lineage_name} Lineage: Shape Space Trajectory', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{lineage_name}_shapespace.png'), dpi=150)
    plt.close()
    
    # Plot 4: Behavioral states (if clustered)
    if 'cluster' in df_results.columns:
        plt.figure(figsize=(8, 8))
        sns.scatterplot(data=df_results, x=np.sqrt(df_results['l1']), 
                       y=np.sqrt(df_results['l2']), hue='cluster', 
                       palette='deep', s=80, alpha=0.7)
        plt.xlabel('Length (L1)', fontsize=12)
        plt.ylabel('Width (L2)', fontsize=12)
        plt.title(f'{lineage_name} Lineage: Behavioral States', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(title='State')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{lineage_name}_states.png'), dpi=150)
        plt.close()
        
    # Save CSV
    df_results.to_csv(os.path.join(output_dir, f'{lineage_name}_analysis.csv'), index=False)
    
    print(f"  Analyzed {len(df_results)} time points, {df_results['num_cells'].max():.0f} max cells")
    return df_results

def compare_lineages(results_dict, output_dir):
    """
    Generate comparison plots across multiple lineages.
    
    Args:
        results_dict: {lineage_name: df_results}
        output_dir: directory to save comparison plots
    """
    print("\nGenerating comparison plots...")
    
    # Plot 1: Overlay of principal axes
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for lineage, df in results_dict.items():
        if df is None or len(df) == 0:
            continue
        axes[0].plot(df['t'], np.sqrt(df['l1']), label=lineage, linewidth=2)
        axes[1].plot(df['t'], np.sqrt(df['l2']), label=lineage, linewidth=2)
        axes[2].plot(df['t'], np.sqrt(df['l3']), label=lineage, linewidth=2)
        
    axes[0].set_title('L1 (Length) Comparison', fontsize=12)
    axes[1].set_title('L2 (Width) Comparison', fontsize=12)
    axes[2].set_title('L3 (Depth) Comparison', fontsize=12)
    
    for ax in axes:
        ax.set_xlabel('Time (frames)')
        ax.set_ylabel('Axis Length (pixels)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_axes.png'), dpi=150)
    plt.close()
    
    # Plot 2: Cell proliferation comparison
    plt.figure(figsize=(10, 6))
    for lineage, df in results_dict.items():
        if df is None or len(df) == 0:
            continue
        plt.plot(df['t'], df['num_cells'], label=lineage, linewidth=2, marker='o', markersize=3)
        
    plt.xlabel('Time (frames)', fontsize=12)
    plt.ylabel('Number of Cells', fontsize=12)
    plt.title('Cell Proliferation Across Lineages', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_proliferation.png'), dpi=150)
    plt.close()
    
    print("  Comparison plots saved")

def main():
    """Main pipeline for lineage-specific shape analysis."""
    # Load labeled tracks
    tracks_path = r"Week2\results\tracks_with_names.csv"
    output_base = r"Week2\results\lineages"
    
    print("Loading labeled tracks...")
    df = pd.read_csv(tracks_path)
    
    # Target lineages
    target_lineages = ['AB', 'P1', 'EMS', 'MS', 'E', 'C']
    
    results_dict = {}
    
    for lineage in target_lineages:
        # Filter cells belonging to this lineage (starts with lineage name)
        df_lineage = df[df['Name'].str.startswith(lineage, na=False)]
        
        if len(df_lineage) == 0:
            print(f"No data for {lineage} lineage")
            continue
            
        # Create output directory
        lineage_dir = os.path.join(output_base, lineage)
        
        # Analyze
        results = analyze_lineage_cloud(df_lineage, lineage, lineage_dir)
        results_dict[lineage] = results
        
    # Generate comparison plots
    if len(results_dict) > 1:
        compare_lineages(results_dict, output_base)
        
    print("\n=== Analysis Complete ===")
    print(f"Analyzed {len(results_dict)} lineages")
    print(f"Results saved to {output_base}")

if __name__ == "__main__":
    main()
