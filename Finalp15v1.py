
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd
import scipy.io

# Load data from the .mat file
mnist_data = scipy.io.loadmat("mnist2.mat")
imgs_train = mnist_data['imgs_train']  # Shape: (20, 20, 60000) or (28, 28, 60000)
labels_train = mnist_data['labels_train']  # Shape: (1, 60000)

# Transpose imgs_train to bring the samples to the first axis
X_full = imgs_train.transpose(2, 0, 1)  # Now shape is (60000, height, width)
y_full = labels_train.flatten()          # Ensure labels are a 1D array

# Print shapes to verify
print(f"Shape of X_full before reshaping: {X_full.shape}")  # Should be (60000, height, width)
print(f"Shape of y_full: {y_full.shape}")                   # Should be (60000,)

# Check if reshaping is needed (if images are in 20x20 or 28x28 format)
image_height, image_width = X_full.shape[1], X_full.shape[2]
num_features = image_height * image_width

if X_full.ndim == 3 and (image_height in [20, 28]) and (image_width in [20, 28]):
    # Reshape to (num_samples, num_features)
    X_full = X_full.reshape(-1, num_features)
    print(f"Shape of X_full after reshaping: {X_full.shape}")  # Should be (60000, num_features)
else:
    # Handle unexpected shapes
    raise ValueError(f"Unexpected shape for X_full: {X_full.shape}")

# Ensure labels are integers
y_full = y_full.astype(int)

# Filter data for specific digits (3, 8, and 9)
classes = [3, 8, 9]
mask = np.isin(y_full, classes)
X, y = X_full[mask], y_full[mask]

# Dataset information
n_samples, n_features = X.shape
n_classes = len(classes)
print(f"Number of samples: {n_samples}, Number of features: {n_features}")
print(f"Number of classes: {n_classes}")

# Center the data by subtracting the mean
X_mean = np.mean(X, axis=0)
X_centered = X - X_mean

# Perform Singular Value Decomposition (SVD) on the centered data
# For the principal components (eigenvectors)
U, S, Vt = svd(X_centered, full_matrices=False)

# Take the top two principal components (Vt contains the right singular vectors)
# Note: Vt is of shape (n_features, n_features)
PCs = Vt[:2].T  # Shape: (n_features, 2)

# Project the data onto the first two principal components
X_projected = X_centered @ PCs  # Shape: (n_samples, 2)

# Visualize the projected data in the PCA space
plt.figure(figsize=(10, 7))

# Configure colors and labels for each class
colors = {3: 'red', 8: 'green', 9: 'blue'}
labels = {3: 'Digit 3', 8: 'Digit 8', 9: 'Digit 9'}

for cls in classes:
    class_mask = y == cls
    plt.scatter(X_projected[class_mask, 0], X_projected[class_mask, 1],
                c=colors[cls], label=labels[cls], alpha=0.5)

# Add plot title, labels, grid, and legend
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Projection of MNIST Digits 3, 8, and 9')
plt.legend()
plt.grid(True)
plt.show()
