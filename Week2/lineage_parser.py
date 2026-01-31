"""
Lineage Parser for C. elegans Embryo Cell Tracking Data

This module parses the ground truth tracking tree (man_track.txt) and assigns
biological names to cells based on C. elegans canonical lineage nomenclature.
"""

import pandas as pd
import numpy as np
from collections import defaultdict, deque

class LineageNode:
    """Represents a single cell in the lineage tree."""
    def __init__(self, cell_id, start_frame, end_frame, parent_id):
        self.cell_id = cell_id
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.parent_id = parent_id
        self.children = []
        self.name = None  # Biological name (e.g., "AB", "P1")
        self.generation = 0
        
    def __repr__(self):
        return f"Cell({self.cell_id}, {self.name}, t={self.start_frame}-{self.end_frame})"

def load_tracking_tree(man_track_path):
    """
    Load the tracking tree from man_track.txt file.
    
    Format: cell_id start_frame end_frame parent_id
    
    Returns:
        dict: {cell_id: LineageNode}
    """
    print(f"Loading tracking tree from {man_track_path}...")
    nodes = {}
    
    with open(man_track_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
                
            cell_id = int(parts[0])
            start_frame = int(parts[1])
            end_frame = int(parts[2])
            parent_id = int(parts[3])
            
            node = LineageNode(cell_id, start_frame, end_frame, parent_id)
            nodes[cell_id] = node
            
    # Build parent-child relationships
    for cell_id, node in nodes.items():
        if node.parent_id != 0:  # Not root
            if node.parent_id in nodes:
                nodes[node.parent_id].children.append(node)
                
    # Calculate generation levels
    root = nodes[1]  # P0 is cell_id=1
    root.generation = 0
    queue = deque([root])
    while queue:
        node = queue.popleft()
        for child in node.children:
            child.generation = node.generation + 1
            queue.append(child)
            
    print(f"Loaded {len(nodes)} cells")
    return nodes

def assign_biological_names(nodes, tracks_df):
    """
    Assign biological names to cells based on C. elegans lineage rules.
    
    Uses spatial information from tracks_df to disambiguate divisions
    (anterior vs posterior daughters).
    
    Args:
        nodes: dict of LineageNode objects
        tracks_df: DataFrame with columns [t, cell_id, x, y, z, volume]
        
    Returns:
        dict: {cell_id: biological_name}
    """
    print("Assigning biological names...")
    
    # Start with P0 (zygote)
    root = nodes[1]
    root.name = "P0"
    
    # Recursive naming function
    def name_daughters(parent_node, parent_name):
        """Recursively name daughter cells."""
        if len(parent_node.children) == 0:
            return
        if len(parent_node.children) > 2:
            print(f"Warning: Cell {parent_node.cell_id} ({parent_name}) has {len(parent_node.children)} children (expected ≤2)")
            
        # Get spatial positions of daughters at birth
        daughters = parent_node.children[:2]  # Take first 2 if more exist
        
        if len(daughters) == 2:
            d1, d2 = daughters
            
            # Get positions at birth frame
            d1_pos = get_cell_position(tracks_df, d1.cell_id, d1.start_frame)
            d2_pos = get_cell_position(tracks_df, d2.cell_id, d2.start_frame)
            
            # Determine anterior/posterior based on Y coordinate (lower Y = anterior in this orientation)
            # Or use principal axis of parent cloud if available
            # For simplicity, use Y-axis
            if d1_pos is not None and d2_pos is not None:
                if d1_pos[1] < d2_pos[1]:  # d1 is anterior (lower Y)
                    anterior, posterior = d1, d2
                else:
                    anterior, posterior = d2, d1
            else:
                # Fallback: use birth order
                anterior, posterior = daughters[0], daughters[1]
                
            # Apply naming rules
            if parent_name == "P0":
                anterior.name = "AB"
                posterior.name = "P1"
            elif parent_name == "P1":
                anterior.name = "EMS"
                posterior.name = "P2"
            elif parent_name == "EMS":
                anterior.name = "MS"
                posterior.name = "E"
            elif parent_name == "P2":
                anterior.name = "C"
                posterior.name = "P3"
            elif parent_name == "P3":
                anterior.name = "D"
                posterior.name = "P4"
            elif parent_name == "AB":
                anterior.name = "ABa"
                posterior.name = "ABp"
            else:
                # Generic naming for further divisions
                anterior.name = parent_name + "a"
                posterior.name = parent_name + "p"
                
            # Recurse
            name_daughters(anterior, anterior.name)
            name_daughters(posterior, posterior.name)
            
        elif len(daughters) == 1:
            # Single daughter (unusual but handle it)
            daughters[0].name = parent_name + ".1"
            name_daughters(daughters[0], daughters[0].name)
            
    # Start recursive naming from P0
    name_daughters(root, root.name)
    
    # Create lookup dict
    name_map = {}
    for cell_id, node in nodes.items():
        if node.name:
            name_map[cell_id] = node.name
        else:
            name_map[cell_id] = f"Unknown_{cell_id}"
            
    print(f"Named {sum(1 for n in name_map.values() if not n.startswith('Unknown'))} cells")
    return name_map

def get_cell_position(tracks_df, cell_id, frame):
    """Get the (x, y, z) position of a cell at a specific frame."""
    row = tracks_df[(tracks_df['cell_id'] == cell_id) & (tracks_df['t'] == frame)]
    if len(row) > 0:
        return (row.iloc[0]['x'], row.iloc[0]['y'], row.iloc[0]['z'])
    return None

def match_to_tracks(nodes, name_map, tracks_csv_path):
    """
    Match biological names to tracks.csv and create labeled DataFrame.
    
    Returns:
        DataFrame with added 'Name' column
    """
    print(f"Loading tracks from {tracks_csv_path}...")
    df = pd.read_csv(tracks_csv_path)
    
    # Add Name column
    df['Name'] = df['cell_id'].map(name_map)
    df['Name'] = df['Name'].fillna('Unknown')
    
    print(f"Matched {len(df[df['Name'] != 'Unknown'])} / {len(df)} rows to named cells")
    return df

def export_lineage_labels(nodes, name_map, output_path):
    """Export lineage labels to CSV."""
    data = []
    for cell_id, node in nodes.items():
        data.append({
            'cell_id': cell_id,
            'Name': name_map.get(cell_id, 'Unknown'),
            'parent_id': node.parent_id,
            'start_frame': node.start_frame,
            'end_frame': node.end_frame,
            'generation': node.generation
        })
        
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Exported lineage labels to {output_path}")
    return df

def main():
    """Main pipeline for lineage parsing."""
    import os
    
    # Paths
    man_track_path = r"Data\Train_Data\Fluo-N3DH-CE\01_GT\TRA\man_track.txt"
    tracks_csv_path = r"tracks.csv"
    output_dir = r"Week2\results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load tracking tree
    nodes = load_tracking_tree(man_track_path)
    
    # Step 2: Load tracks for spatial info
    tracks_df = pd.read_csv(tracks_csv_path)
    
    # Step 3: Assign biological names
    name_map = assign_biological_names(nodes, tracks_df)
    
    # Step 4: Export lineage labels
    lineage_df = export_lineage_labels(nodes, name_map, 
                                       os.path.join(output_dir, "lineage_labels.csv"))
    
    # Step 5: Create labeled tracks CSV
    labeled_tracks = match_to_tracks(nodes, name_map, tracks_csv_path)
    labeled_tracks.to_csv(os.path.join(output_dir, "tracks_with_names.csv"), index=False)
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Total cells: {len(nodes)}")
    print(f"Named cells: {sum(1 for n in name_map.values() if not n.startswith('Unknown'))}")
    print("\nKey lineages:")
    for key_name in ['P0', 'AB', 'P1', 'EMS', 'MS', 'E', 'P2', 'C', 'P3', 'ABa', 'ABp']:
        count = sum(1 for n in name_map.values() if n == key_name)
        if count > 0:
            print(f"  {key_name}: {count} cell(s)")

if __name__ == "__main__":
    main()
