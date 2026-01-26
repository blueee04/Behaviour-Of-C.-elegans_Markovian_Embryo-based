import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.exposure import rescale_intensity
import argparse

def visualize_raw_data(data_dir, output_prefix="raw_data_vis"):
    print(f"Searching for TIFFs in {data_dir}...")
    files = sorted(glob.glob(os.path.join(data_dir, "*.tif")))
    
    if not files:
        print("No .tif files found.")
        return

    print(f"Found {len(files)} files.")
    
    # Select 3 representative frames: Start, Middle, End
    indices = [0, len(files)//2, len(files)-1]
    selected_files = [files[i] for i in indices]
    
    # storage for montage
    mips = []
    
    for i, file_path in zip(indices, selected_files):
        print(f"Processing frame {i}: {os.path.basename(file_path)}")
        try:
            # Load 3D volume (Z, Y, X)
            img = imread(file_path)
            print(f"  Shape: {img.shape}, dtype: {img.dtype}, Range: {img.min()}-{img.max()}")
            
            # Maximum Intensity Projection along Z (axis 0)
            mip = np.max(img, axis=0)
            
            # Contrast stretching for better visualization
            # robust min/max (start at 2nd percentile to ignore background noise causing black crush)
            p2, p98 = np.percentile(mip, (2, 98))
            mip_rescaled = rescale_intensity(mip, in_range=(p2, p98))
            
            mips.append((i, mip_rescaled))
            
            # Save individual MIP
            plt.figure(figsize=(8, 8))
            plt.imshow(mip_rescaled, cmap='gray')
            plt.title(f"MIP Frame {i}\n{os.path.basename(file_path)}")
            plt.axis('off')
            outfile = f"{output_prefix}_frame_{i:03d}_mip.png"
            plt.savefig(outfile, bbox_inches='tight', pad_inches=0)
            plt.close()
            print(f"  Saved {outfile}")
            
        except Exception as e:
            print(f"  Error reading {file_path}: {e}")

    # Create Montage
    if mips:
        print("Creating montage...")
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for ax, (t, mip) in zip(axes, mips):
            ax.imshow(mip, cmap='gray')
            ax.set_title(f"Frame {t}")
            ax.axis('off')
            
        plt.tight_layout()
        outfile = f"{output_prefix}_montage.png"
        plt.savefig(outfile)
        print(f"Saved montage to {outfile}")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Default to the path we found earlier
    default_path = r"d:\Github\Behaviour-Of-C.-elegans_Markovian_Embryo-based\Data\Train_Data\Fluo-N3DH-CE\01"
    parser.add_argument("--data_dir", type=str, default=default_path)
    parser.add_argument("--output_prefix", type=str, default="raw_data")
    args = parser.parse_args()
    
    visualize_raw_data(args.data_dir, args.output_prefix)
