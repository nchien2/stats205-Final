import numpy as np

def rbf_kernel_dist(x, y, gamma):
    return 1 - np.exp(- gamma * ((x - y) ** 2).sum())

def poly_kernel_dist(x, y, gamma, r=0., d=3):
    Kxx = (r + gamma * (x ** 2).sum()) ** d
    Kyy = (r + gamma * (y ** 2).sum()) ** d
    Kxy = (r + gamma * np.dot(x, y)) ** d
    return Kxx + Kyy - 2 * Kxy

def sigmoid_kernel_dist(x, y, gamma, r=0.):
    Kxx = np.tanh(r + gamma * (x ** 2).sum())
    Kyy = np.tanh(r + gamma * (y ** 2).sum())
    Kxy = np.tanh(r + gamma * np.dot(x, y))
    return Kxx + Kyy - 2 * Kxy