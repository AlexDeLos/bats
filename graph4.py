import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Data preparation
data = {
    'Layers': [4, 9, 4, 9, 4, 9],
    'Datasets': ['MNIST', 'MNIST', 'EMNIST', 'EMNIST', 'Fashion', 'Fashion'],
    'No_Delay': [97.9, 97.2, 88.2, 78.3, 90.0, 87.6],
    'Delay': [87.7, 96.4, 81.8, 79.6, 89.7, 85.8]
}

df = pd.DataFrame(data)

# Plotting
bar_width = 0.35
index = np.arange(len(df))

plt.figure(figsize=(10, 6))

# Bars for No Delay
plt.bar(index, df['No_Delay'], bar_width, label='No Delay')

# Bars for Delay
plt.bar(index + bar_width, df['Delay'], bar_width, label='Delay')

# Labels and Title
plt.xlabel('Layers and Datasets')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison: No Delay vs Delay')

# Adding x-tick labels with layers and datasets
plt.xticks(index + bar_width / 2, df['Layers'].astype(str) + ' - ' + df['Datasets'])

plt.legend()
plt.grid(True, axis='y')
plt.show()
