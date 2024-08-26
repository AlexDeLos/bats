import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data preparation
data = {
    'Layers': [3, 4, 3, 4, 3, 4],
    'Datasets': ['MNIST', 'MNIST', 'EMNIST', 'EMNIST', 'Fashion', 'Fashion'],
    'No_residual': [96.6, None, 71.6, None, 84.3, None],
    'Average': [98.4, 98.0, 72.0, 80.0, 87.1, 87.8],
    'Append': [97.2, 98.0, 71.9, 79.6, 85.8, 87.7]
}

df = pd.DataFrame(data)

# Plotting
layers = df['Layers'].unique()
datasets = df['Datasets'].unique()
bar_width = 0.25

plt.figure(figsize=(12, 8))

for i, dataset in enumerate(datasets):
    # Filter data by dataset
    df_subset = df[df['Datasets'] == dataset]
    
    # Positions of the bars on the x-axis
    r1 = np.arange(len(layers)) + i*(bar_width*4)
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    # Create bars
    plt.bar(r1, df_subset['No_residual'], color='blue', width=bar_width, edgecolor='grey', label=f'No Residual ({dataset})')
    plt.bar(r2, df_subset['Average'], color='orange', width=bar_width, edgecolor='grey', label=f'Average ({dataset})')
    plt.bar(r3, df_subset['Append'], color='green', width=bar_width, edgecolor='grey', label=f'Append ({dataset})')

# Set y-axis limits
plt.ylim(60, 100)

# Add labels with larger font sizes
plt.xlabel('Layers', fontweight='bold', fontsize=14)
plt.ylabel('Accuracy', fontweight='bold', fontsize=14)
plt.xticks([r + bar_width for r in range(len(layers))], ['3', '4'], fontsize=12)
plt.yticks(fontsize=12)

# Add title and legend with larger font size
plt.title('Accuracy Comparison for 3-Layer and 4-Layer Models', fontsize=16)
plt.legend(fontsize=10)

# Display the plot
plt.show()
