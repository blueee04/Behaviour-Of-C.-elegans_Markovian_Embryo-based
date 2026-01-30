import os
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def parse_man_track(filepath):
    """
    Parses man_track.txt file.
    Format usually: ID, Start, End, Parent
    But we observed: ID Start End Parent
    """
    tracks = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = list(map(int, line.strip().split()))
            if len(parts) == 4:
                cell_id, start, end, parent = parts
                tracks[cell_id] = {'start': start, 'end': end, 'parent': parent}
    return tracks

def build_lineage_graph(tracks):
    G = nx.DiGraph()
    for cid, data in tracks.items():
        G.add_node(cid, **data)
        if data['parent'] != 0:
            G.add_edge(data['parent'], cid)
    return G

def get_subtree_size(G, root):
    return len(nx.descendants(G, root)) + 1

def label_cell_types(G, root_id):
    """
    Heuristic labeling based on subtree size for C. elegans.
    P0 -> AB (larger), P1 (smaller)
    P1 -> EMS (larger), P2 (smaller)
    """
    labels = {root_id: 'P0'}
    
    # Queue for BFS: (node_id, assigned_name)
    queue = [(root_id, 'P0')]
    
    while queue:
        pid, name = queue.pop(0)
        daughters = list(G.successors(pid))
        
        if len(daughters) == 2:
            d1, d2 = daughters
            size1 = get_subtree_size(G, d1)
            size2 = get_subtree_size(G, d2)
            
            # Sort by size (descending)
            # Tuple: (size, id)
            sorted_daughters = sorted([(size1, d1), (size2, d2)], key=lambda x: x[0], reverse=True)
            big_d, small_d = sorted_daughters[0][1], sorted_daughters[1][1]
            
            new_name_big = f"{name}_big"
            new_name_small = f"{name}_small"
            
            # Rule based naming
            if name == 'P0':
                labels[big_d] = 'AB'
                labels[small_d] = 'P1'
                queue.append((big_d, 'AB'))
                queue.append((small_d, 'P1'))
                
            elif name == 'P1':
                labels[big_d] = 'EMS'
                labels[small_d] = 'P2'
                queue.append((big_d, 'EMS'))
                queue.append((small_d, 'P2'))
            
            elif name == 'EMS':
                # MS typically larger subtree than E? Or similar.
                # E produces 20 cells, MS produces 80. So MS is larger.
                labels[big_d] = 'MS'
                labels[small_d] = 'E'
                queue.append((big_d, 'MS'))
                queue.append((small_d, 'E'))
                
            elif name == 'P2':
                # C (muscle) vs P3 (germ)
                # C (47 cells) vs P3 (lots of descendants... germline proliferation).
                # Actually P3 -> P4 -> Z2/Z3 is small embryo-wise unless L1.
                # In early embryo, C might be larger.
                # Let's map generically for now or use 'C' and 'P3'
                labels[big_d] = 'C'
                labels[small_d] = 'P3'
                queue.append((big_d, 'C'))
                queue.append((small_d, 'P3'))
            
            elif name == 'AB':
                # ABa vs ABp. Sizes are very similar (ABa=ABp essentially).
                # Cannot distinguish by size easily.
                labels[big_d] = 'AB_1'
                labels[small_d] = 'AB_2'
                queue.append((big_d, 'AB_1'))
                queue.append((small_d, 'AB_2'))
            
            else:
                # Default recursive naming for others
                labels[big_d] = f"{name}a" # arbitrarily assign 'a' to big
                labels[small_d] = f"{name}p"
                queue.append((big_d, labels[big_d]))
                queue.append((small_d, labels[small_d]))
                
        elif len(daughters) == 1:
            # Just pass the name down or append 'x'
            labels[daughters[0]] = f"{name}'"
            queue.append((daughters[0], labels[daughters[0]]))
            
    return labels

def analyze_dataset(path, dataset_name):
    tracks = parse_man_track(path)
    G = build_lineage_graph(tracks)
    
    # Find root (node with parent 0 or indegree 0)
    roots = [n for n, d in G.in_degree() if d == 0]
    # Filter roots that are actually in the track dict
    # (Node 0 is virtual root)
    roots = [r for r in roots if r in tracks]
    
    # Assume single embryo, single root P0. If multiple, take the one with biggest subtree.
    main_root = max(roots, key=lambda r: get_subtree_size(G, r))
    
    labels = label_cell_types(G, main_root)
    
    # Collect metrics
    data = []
    for cid in G.nodes():
        if cid not in tracks: continue
        name = labels.get(cid, "Unknown")
        duration = tracks[cid]['end'] - tracks[cid]['start'] + 1
        data.append({
            'CellID': cid,
            'Name': name,
            'Duration': duration,
            'Start': tracks[cid]['start'],
            'End': tracks[cid]['end'],
            'Dataset': dataset_name
        })
        
    return pd.DataFrame(data), G, labels

def main():
    base_path = r"d:\Github\Behaviour-Of-C.-elegans_Markovian_Embryo-based\Data\Train_Data\Fluo-N3DH-CE"
    path_01 = os.path.join(base_path, "01_GT", "TRA", "man_track.txt")
    path_02 = os.path.join(base_path, "02_GT", "TRA", "man_track.txt")
    
    df1, G1, labels1 = analyze_dataset(path_01, "01")
    df2, G2, labels2 = analyze_dataset(path_02, "02")
    
    # Combine
    full_df = pd.concat([df1, df2])
    
    output_dir = r"d:\Github\Behaviour-Of-C.-elegans_Markovian_Embryo-based\Week_2\results"
    full_df.to_csv(os.path.join(output_dir, "lineage_with_types.csv"), index=False)
    
    # Comparison Report
    # Pivot on Name to compare Duration
    
    # Filter for standard names
    target_names = ['P0', 'AB', 'P1', 'EMS', 'P2', 'MS', 'E', 'C', 'P3']
    report_lines = ["# Dataset Comparison Report\n"]
    report_lines.append(f"## Dataset 01 Summary")
    report_lines.append(f"- Total Tracked Cells: {len(df1)}")
    report_lines.append(f"- Max Frame: {df1['End'].max()}")
    
    report_lines.append(f"\n## Dataset 02 Summary")
    report_lines.append(f"- Total Tracked Cells: {len(df2)}")
    report_lines.append(f"- Max Frame: {df2['End'].max()}")
    
    report_lines.append("\n## Cell Cycle Duration Comparison (Frames)")
    report_lines.append("| Cell | Duration (01) | Duration (02) | Difference |")
    report_lines.append("|---|---|---|---|")
    
    for name in target_names:
        row1 = df1[df1['Name'] == name]
        row2 = df2[df2['Name'] == name]
        
        d1 = row1['Duration'].values[0] if not row1.empty else "-"
        d2 = row2['Duration'].values[0] if not row2.empty else "-"
        
        diff = "-"
        if d1 != "-" and d2 != "-":
            diff = d1 - d2
            
        report_lines.append(f"| {name} | {d1} | {d2} | {diff} |")
        
    with open(os.path.join(output_dir, "comparison_report.md"), "w") as f:
        f.write("\n".join(report_lines))
        
    print("Analysis complete. Results saved to Week_2/results.")

if __name__ == "__main__":
    main()
