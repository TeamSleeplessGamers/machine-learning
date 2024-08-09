# seaborn.py
import seaborn as sns
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

def generate_heatmap(df, heatmap_path):
    pivot_table = df.pivot_table(index='day', columns='time', aggfunc='size', fill_value=0)
    
    if pivot_table.empty:
        raise ValueError("Pivot table is empty, cannot generate heatmap.")
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, fmt='d', cmap='YlGnBu', linewidths=0.5)
    plt.title('Heatmap of Streamer Activity')
    plt.savefig(heatmap_path)
    plt.close()


