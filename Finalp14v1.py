# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky, eigh
from numpy.linalg import inv
import scipy.io

# Load data from the .mat file
mnist_data = scipy.io.loadmat("mnist2.mat")
imgs_train = mnist_data['imgs_train']  # Shape: (20, 20, 60000)
imgs_test = mnist_data['imgs_test']
labels_train = mnist_data['labels_train']  # Shape: (1, 60000)
labels_test = mnist_data['labels_test']

# Transpose imgs_train to bring the samples to the first axis
X_full = imgs_train.transpose(2, 0, 1)  # Now shape is (60000, 20, 20)
y_full = labels_train.flatten()          # Ensure labels are a 1D array

# Print shapes to verify
print(f"Shape of X_full before reshaping: {X_full.shape}")  # Should be (60000, 20, 20)
print(f"Shape of y_full: {y_full.shape}")                   # Should be (60000,)

# Check if reshaping is needed (if images are in 20x20 format)
if X_full.ndim == 3 and X_full.shape[1:] == (20, 20):
    # Reshape to (num_samples, 400)
    X_full = X_full.reshape(-1, 20 * 20)
    print(f"Shape of X_full after reshaping: {X_full.shape}")  # Should be (60000, 400)
elif X_full.ndim == 2 and X_full.shape[1] == 400:
    # Data is already in the correct shape
    print("No reshaping needed for X_full.")
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

# Compute the overall mean of the dataset
m = np.mean(X, axis=0)

# Compute class-wise mean vectors and sample counts
m_i, n_i = {}, {}
for cls in classes:
    class_mask = y == cls
    X_i = X[class_mask]
    n_i[cls] = X_i.shape[0]
    m_i[cls] = np.mean(X_i, axis=0)
    print(f"Class {cls}: Number of samples = {n_i[cls]}")

# Compute the within-class scatter matrix (S_w)
S_w = np.zeros((n_features, n_features))
for cls in classes:
    class_mask = y == cls
    X_i = X[class_mask]
    mean_i = m_i[cls]
    X_centered = X_i - mean_i
    S_w += X_centered.T @ X_centered

# Regularize S_w to ensure it's positive definite (if necessary)
epsilon = 1e-6
S_w += epsilon * np.eye(n_features)

# Compute the between-class scatter matrix (S_b)
S_b = np.zeros((n_features, n_features))
for cls in classes:
    mean_diff = (m_i[cls] - m).reshape(-1, 1)
    S_b += n_i[cls] * (mean_diff @ mean_diff.T)

# Perform Cholesky decomposition on S_w
L = cholesky(S_w, lower=True)

# Compute the inverse of the Cholesky factor
L_inv = inv(L)

# Transform the generalized eigenvalue problem into a standard one
A = L_inv.T @ S_b @ L_inv

# Solve the eigenvalue problem for matrix A
eigvals, eigvecs = eigh(A)

# Sort eigenvalues and eigenvectors in descending order
sorted_indices = np.argsort(eigvals)[::-1]
eigvals = eigvals[sorted_indices]
eigvecs = eigvecs[:, sorted_indices]

# Take the top two eigenvectors for dimensionality reduction
Y = eigvecs[:, :2]  # Top two eigenvectors for 2D projection

# Compute the projection matrix W
W = L_inv @ Y

# Project the data onto the new LDA space
X_projected = X @ W

# Visualize the projected data in the reduced LDA space
plt.figure(figsize=(10, 7))

# Configure colors and labels for each class
colors = {3: 'red', 8: 'green', 9: 'blue'}
labels = {3: 'Digit 3', 8: 'Digit 8', 9: 'Digit 9'}

for cls in classes:
    class_mask = y == cls
    plt.scatter(X_projected[class_mask, 0], X_projected[class_mask, 1],
                c=colors[cls], label=labels[cls], alpha=0.5)

# Add plot title, labels, grid, and legend
plt.xlabel('LD1')  # First Linear Discriminant
plt.ylabel('LD2')  # Second Linear Discriminant
plt.title('LDA Projection of MNIST Digits 3, 8, and 9')
plt.legend()
plt.grid(True)
plt.show()
