import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

def inspect_activity():
    df = pd.read_csv('tracks.csv')
    times = sorted(df['t'].unique())
    results = []
    
    for t in times:
        cells = df[df['t'] == t]
        if len(cells) < 3:
            continue
            
        P = cells[['x', 'y', 'z']].values
        P_centered = P - np.mean(P, axis=0)
        tensor = np.dot(P_centered.T, P_centered) / len(P)
        eigvals = np.linalg.eigvalsh(tensor)
        # ascending: l1=max is last
        results.append({
            't': t,
            'l1': eigvals[-1], # Length
            'l2': eigvals[-2]  # Width
        })
        
    res_df = pd.DataFrame(results)
    l1 = np.sqrt(res_df['l1'])
    l2 = np.sqrt(res_df['l2'])
    
    # Detrend
    l1_trend = savgol_filter(l1, 21, 3)
    l2_trend = savgol_filter(l2, 21, 3)
    
    activity = pd.Series(l1 - l1_trend).rolling(5).std().fillna(0) + \
               pd.Series(l2 - l2_trend).rolling(5).std().fillna(0)
               
    print(f"Max Activity: {activity.max():.4f}")
    
    # Print periods of low activity
    threshold = 0.2 * activity.max()
    low_activity = res_df[activity < threshold]
    
    print("\nLow Activity Frames (< 20% of max):")
    print(low_activity['t'].values)
    
    # Print the last few values to see if it drops at end
    print("\nLast 10 Activity Values:")
    print(activity.tail(10).values)

inspect_activity()
