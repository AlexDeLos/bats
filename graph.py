import pandas as pd
import matplotlib.pyplot as plt

# Data preparation
data = {
    'Layers': [1, 4, 9, 13, 1, 4, 5, 9, 13, 1, 4, 9, 13],
    'Datasets': ['MNIST']*4 + ['EMNIST']*5 + ['Fashion']*4,
    'No_residual': [98.2, 97.6, None, None, 75.3, 82.6, 76.4, None, None, 88.8, 88.6, None, None],
    'Average': [None, 97.8, 96.9, 95.1, None, 78.6, 80.1, 78.7, 71.6, None, 89.9, 87.2, 75.8],
    'Append': [None, 97.2, 97.0, 97.1, None, 81.3, 82.7, 83.1, 75.5, None, 88.8, 80.7, 77.9]
}

# train accuracy
data = {
    'Layers': [1, 4, 9, 13, 1, 4, 5, 9, 13, 1, 4, 9, 13],
    'Datasets': ['MNIST']*4 + ['EMNIST']*5 + ['Fashion']*4,
    'No_residual': [98.9, 99.6, None, None, 76.7, 88.4, 80.1, None, None, 91.6, 99.1, None, None],
    'Average': [None, 99.8, 99.2, 98.0, None, 80.2, 81.4, 79.2, 73.0, None, 98.5, 98.9, 87.2],
    'Append': [None, 99.8, 99.8, 99.8, None, 85.5, 86.2, 88.5, 80.9, None, 99.4, 88.7, 87.2]
}


df = pd.DataFrame(data)

# Plotting
datasets = df['Datasets'].unique()

plt.figure(figsize=(12, 8))

for dataset in datasets:
    subset = df[df['Datasets'] == dataset]
    plt.plot(subset['Layers'], subset['No_residual'], label=f'{dataset} - No Residual', marker='o')
    plt.plot(subset['Layers'], subset['Average'], label=f'{dataset} - Average', marker='o')
    plt.plot(subset['Layers'], subset['Append'], label=f'{dataset} - Append', marker='o')

plt.xlabel('Layers')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Layers for Different Datasets and Models')
plt.legend()
plt.grid(True)
plt.show()
