import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

Rhoop = 3  # the radius of the hoop
r0 = 1     # the equilibrium length of each spring
kappa = 1  # the spring constant
Nnodes = 21
A = np.zeros((Nnodes, Nnodes), dtype=int)  # adjacency matrix

# vertical springs
for k in range(3):
    A[k, k + 4] = 1
for k in range(4, 7):
    A[k, k + 5] = 1
for k in range(9, 12):
    A[k, k + 5] = 1
for k in range(14, 17):
    A[k, k + 4] = 1
# horizontal springs
for k in range(3, 7):
    A[k, k + 1] = 1
for k in range(8, 12):
    A[k, k + 1] = 1
for k in range(13, 17):
    A[k, k + 1] = 1

Asymm = A + np.transpose(A)  # Symmetrize the adjacency matrix
ind_hoop = [0, 3, 8, 13, 18, 19, 20, 17, 12, 7, 2, 1]  # Indices of nodes on the hoop (red nodes)
Nhoop = np.size(ind_hoop)
ind_free = [4, 5, 6, 9, 10, 11, 14, 15, 16]  # Indices of free nodes (black nodes)
Nfree = np.size(ind_free)
springs = np.array(np.nonzero(A))  # springs list
Nsprings = np.size(springs, axis=1)

# Initial angles and position for the hoop nodes uniformly distributed around 2*pi starting from theta0
theta0 = 2 * np.pi / 3
theta = theta0 + np.linspace(0, 2 * np.pi, Nhoop + 1)
theta = np.delete(theta, -1)
pos = np.zeros((Nnodes, 2))
pos[ind_hoop, 0] = Rhoop * np.cos(theta)
pos[ind_hoop, 1] = Rhoop * np.sin(theta)
pos[ind_free, 0] = np.array([-1., 0., 1., -1., 0., 1., -1., 0., 1.])
pos[ind_free, 1] = np.array([1., 1., 1., 0., 0., 0., -1., -1., -1.])

# Initialize the vector of parameters to be optimized
vec = np.concatenate((theta, pos[ind_free, 0], pos[ind_free, 1]))

def vec_to_pos(vec):
    theta = vec[:Nhoop]  # Split the input vector vec into theta and positions
    pos = np.zeros((Nnodes, 2))
    pos[ind_hoop, 0] = Rhoop * np.cos(theta)
    pos[ind_hoop, 1] = Rhoop * np.sin(theta)
    pos[ind_free, 0] = vec[Nhoop:Nhoop+Nfree]   # positions of the free nodes (x-components)
    pos[ind_free, 1] = vec[Nhoop+Nfree:Nhoop+2*Nfree]  # positions of the free nodes (y-components)
    return theta, pos

def Energy(theta, pos, springs, r0, kappa):
    Nsprings = springs.shape[1]
    E = 0.
    for k in range(Nsprings):
        j0 = springs[0, k]
        j1 = springs[1, k]
        rvec = pos[j0, :] - pos[j1, :]
        rvec_length = np.linalg.norm(rvec)
        E += 0.5 * kappa * (rvec_length - r0) ** 2
    return E

def compute_gradient(theta, pos, Asymm, r0, kappa, R, ind_hoop, ind_free):
    Nhoop = len(ind_hoop)
    g_hoop = np.zeros((Nhoop,))  # gradient w.r.t the angles of the hoop nodes
    Nfree = len(ind_free)
    g_free = np.zeros((Nfree, 2))  # gradient w.r.t x- and y-components of free nodes
    # Gradient with respect to hoop angles
    for idx_hoop, node in enumerate(ind_hoop):
        adj_nodes = np.nonzero(Asymm[node, :])[0]
        for adj_node in adj_nodes:
            rvec = pos[node, :] - pos[adj_node, :]
            rvec_length = np.linalg.norm(rvec)
            if rvec_length == 0:
                continue
            # Derivative of energy with respect to position of hoop node
            dE_dpos = kappa * (rvec_length - r0) * (rvec / rvec_length)
            # Position derivative with respect to theta
            dpos_dtheta = R * np.array([-np.sin(theta[idx_hoop]), np.cos(theta[idx_hoop])])
            # Chain rule
            g_hoop[idx_hoop] += np.dot(dE_dpos, dpos_dtheta)
    # Gradient with respect to free nodes (x and y positions)
    for idx_free, node in enumerate(ind_free):
        adj_nodes = np.nonzero(Asymm[node, :])[0]
        for adj_node in adj_nodes:
            rvec = pos[node, :] - pos[adj_node, :]
            rvec_length = np.linalg.norm(rvec)
            if rvec_length == 0:
                continue
            # Derivative of energy with respect to position of free node
            dE_dpos = kappa * (rvec_length - r0) * (rvec / rvec_length)
            g_free[idx_free, :] += dE_dpos
    grad = np.concatenate([g_hoop, g_free[:, 0], g_free[:, 1]])  # Combine gradient components
    return grad

def func(vec):
    theta, pos = vec_to_pos(vec)
    return Energy(theta, pos, springs, r0, kappa)

def gradient(vec):
    theta, pos = vec_to_pos(vec)
    grad = compute_gradient(theta, pos, Asymm, r0, kappa, Rhoop, ind_hoop, ind_free)
    return grad

def draw_spring_system(pos, springs, R, ind_hoop, ind_free, title='Spring System'):
    plt.figure()
    # Plot the hoop
    theta_plot = np.linspace(0, 2 * np.pi, 100)
    x_hoop = R * np.cos(theta_plot)
    y_hoop = R * np.sin(theta_plot)
    plt.plot(x_hoop, y_hoop, 'k-', linewidth=0.5)
    # Plot the springs
    for k in range(springs.shape[1]):
        j0 = springs[0, k]
        j1 = springs[1, k]
        x = [pos[j0, 0], pos[j1, 0]]
        y = [pos[j0, 1], pos[j1, 1]]
        plt.plot(x, y, 'b-', linewidth=1)
    # Plot the nodes on the hoop
    plt.plot(pos[ind_hoop, 0], pos[ind_hoop, 1], 'ro', label='Hoop Nodes')
    # Plot the free nodes
    plt.plot(pos[ind_free, 0], pos[ind_free, 1], 'ko', label='Free Nodes')
    plt.axis('equal')
    plt.title(title)
    plt.legend()
    plt.show()

# First optimization method: BFGS
energies_bfgs = []
grad_norms_bfgs = []

def callback_bfgs(vec):
    E = func(vec)
    grad = gradient(vec)
    grad_norm = np.linalg.norm(grad)
    energies_bfgs.append(E)
    grad_norms_bfgs.append(grad_norm)

res_bfgs = minimize(func, vec, method='BFGS', jac=gradient,
                    callback=callback_bfgs, options={'disp': True})

# Second optimization method: Conjugate Gradient (CG)
energies_cg = []
grad_norms_cg = []

def callback_cg(vec):
    E = func(vec)
    grad = gradient(vec)
    grad_norm = np.linalg.norm(grad)
    energies_cg.append(E)
    grad_norms_cg.append(grad_norm)

res_cg = minimize(func, vec, method='CG', jac=gradient,
                  callback=callback_cg, options={'disp': True})

# Plot the spring energy and the norm of its gradient versus the iteration number for each method
plt.figure()
plt.plot(energies_bfgs, label='BFGS Energy')
plt.plot(energies_cg, label='CG Energy')
plt.xlabel('Iteration')
plt.ylabel('Energy')
plt.title('Spring Energy vs Iteration Number')
plt.legend()
plt.show()

plt.figure()
plt.plot(grad_norms_bfgs, label='BFGS Gradient Norm')
plt.plot(grad_norms_cg, label='CG Gradient Norm')
plt.xlabel('Iteration')
plt.ylabel('Gradient Norm')
plt.title('Gradient Norm vs Iteration Number')
plt.legend()
plt.show()

# Plot the resulting view of the spring system stretched on the hoop for each method
theta_opt_bfgs, pos_opt_bfgs = vec_to_pos(res_bfgs.x)
draw_spring_system(pos_opt_bfgs, springs, Rhoop, ind_hoop, ind_free, title='Spring System - BFGS Optimization')

theta_opt_cg, pos_opt_cg = vec_to_pos(res_cg.x)
draw_spring_system(pos_opt_cg, springs, Rhoop, ind_hoop, ind_free, title='Spring System - CG Optimization')

# Print the positions of the nodes, the resulting energy, and the norm of the gradient for each method
print("BFGS Optimization Results:")
print("Positions of Nodes:")
print(pos_opt_bfgs)
print("Resulting Energy: {:.6f}".format(func(res_bfgs.x)))
print("Norm of Gradient: {:.6f}".format(np.linalg.norm(gradient(res_bfgs.x))))
print("===================================")
print("CG Optimization Results:")
print("Positions of Nodes:")
print(pos_opt_cg)
print("Resulting Energy: {:.6f}".format(func(res_cg.x)))
print("Norm of Gradient: {:.6f}".format(np.linalg.norm(gradient(res_cg.x))))
