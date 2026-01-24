import os
import glob
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.measure import regionprops
from tqdm import tqdm
import argparse

def extract_tracks(data_dir, output_file, limit=None):
    """
    Extracts cell centroids from Cell Tracking Challenge TIFF masks.
    """
    # Search for man_track*.tif files
    print(f"Searching for files in {data_dir}...")
    files = sorted(glob.glob(os.path.join(data_dir, "man_track*.tif")))
    
    if not files:
        print("No man_track*.tif files found!")
        return

    if limit:
        files = files[:limit]

    tracks = []
    
    print(f"Processing {len(files)} frames...")
    
    for t, file_path in enumerate(tqdm(files)):
        # Inspect filename to get actual time index if needed, but enumeration is usually fine for CTC
        # filename format: man_trackXXX.tif
        try:
            # Load image
            img = imread(file_path)
            print(f"Loaded {file_path}, shape: {img.shape}, dtype: {img.dtype}, max: {img.max()}")
            
            # regionprops gives us properties of labeled regions
            # The image contains integer labels for each cell
            props = regionprops(img)
            print(f"Found {len(props)} regions.")
            
            for region in props:
                z, y, x = region.centroid
                label_id = region.label
                
                tracks.append({
                    't': t,
                    'cell_id': label_id,
                    'z': z,
                    'y': y,
                    'x': x,
                    'volume': region.area # Number of pixels
                })
                
        except Exception as e:
            print(f"Error processing frame {t} ({file_path}): {e}")

    # Convert to DataFrame
    df = pd.DataFrame(tracks)
    
    # Save to CSV
    print(f"Saving tracks to {output_file}...")
    df.to_csv(output_file, index=False)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract cell tracks from CTC TIFFs.")
    parser.add_argument("--input_dir", type=str, default=r"d:\Github\Behaviour-Of-C.-elegans_Markovian_Embryo-based\Data\Train_Data\Fluo-N3DH-CE\01_GT\TRA", help="Directory containing man_track*.tif files")
    parser.add_argument("--output_file", type=str, default="tracks.csv", help="Output CSV file path")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of frames to process (for testing)")
    
    args = parser.parse_args()
    
    extract_tracks(args.input_dir, args.output_file, args.limit)
