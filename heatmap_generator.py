# seaborn.py
import seaborn as sns
import pandas as pd
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

def generate_heatmap(data, filename):
    # Convert data to DataFrame
    df = pd.DataFrame(data)
    df.set_index('username', inplace=True)

    # Create a heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df, cmap='Blues', annot=True)
    plt.title("Dummy User Online Activity Heatmap")
    plt.xlabel("Hour of Day")
    plt.ylabel("Username")

    # Save the heatmap to a file
    plt.savefig(filename, format='png')
    plt.close()

