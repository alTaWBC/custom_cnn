import numpy as np
import random


def convolution(matrix, kernel, stride):
    x = (matrix.shape[0] - kernel.shape[0]) // stride[0] + 1
    y = (matrix.shape[1] - kernel.shape[1]) // stride[1] + 1

    convolution = np.zeros((x, y))
    for i in range(x):
        for j in range(y):
            start_x, end_x = i * \
                stride[0], i * stride[0] + kernel.shape[0]
            start_y, end_y = j * \
                stride[1], j * stride[1] + kernel.shape[1]
            window = matrix[start_x: end_x, start_y: end_y]

            convolution[i, j] = np.sum(window * kernel)
    return convolution


def convolution2(matrix, kernel, stride):

    x = (matrix.shape[0] - kernel.shape[0]) // stride[0] + 1
    y = (matrix.shape[1] - kernel.shape[1]) // stride[1] + 1
    z = (matrix.shape[2] - kernel.shape[2]) // stride[2] + 1
    convolution = np.zeros((x, y, z))
    for i in range(x):
        for j in range(y):
            for k in range(z):
                start_x, end_x = i * stride[0], i * stride[0] + kernel.shape[0]
                start_y, end_y = j * stride[1], j * stride[1] + kernel.shape[1]
                start_z, end_z = k * stride[2], k * stride[2] + kernel.shape[2]

                window = matrix[start_x: end_x, start_y: end_y, start_z: end_z]

                convolution[i, j, k] = np.sum(
                    window * kernel)
    return convolution


def batch_normalization(batch):
    epsilon = 0.001
    gamma = 1.0
    beta = 0.0
    moving_mean = 0.0
    moving_variance = 1.0
    return (batch - moving_mean) / (moving_variance + epsilon) * gamma + beta


def activation(matrix):
    return np.maximum(matrix, 0)


def max_norm(matrix):
    return np.minimum(matrix, 2)


def pooling(matrix, max_pool):
    x, y = matrix.shape[0] - 1, matrix.shape[1] - 1
    pooling = np.zeros((x, y))

    for i in range(x):
        for j in range(y):
            start_x, end_x = i, i + max_pool[0]
            start_y, end_y = j, j + max_pool[1]
            window = matrix[start_x: end_x, start_y: end_y]
            value = np.max(window)
            pooling[i, j] = value
    return pooling


def dense(input_matrix, neuron_matrix):
    return np.dot(input_matrix, neuron_matrix.T)


def softmax(matrix):
    exp_matrix = np.exp(matrix)
    return exp_matrix / np.sum(exp_matrix)


if __name__ == '__main__':
    kernel1 = np.ones((10, 2))
    kernel2 = np.ones((10, 2))
    stride1 = (2, 1)
    stride2 = (1, 1)

    windowPool = (2, 2)

    convolution1 = convolution(
        np.ones((80, 9)), kernel1, stride1)

    pooling1 = pooling(convolution1, windowPool)

    convolution2 = convolution(pooling1, kernel2, stride2)
    pooling2 = pooling(convolution2, windowPool)
    flatten = pooling2.flatten()

    dense1 = dense(flatten, np.ones((flatten.shape)))

    print(convolution1.shape)
    print(pooling1.shape)
    print(convolution2.shape)
    print(pooling2.shape)
    print(dense1.shape)

    print(batch_normalization(np.ones((10, 36, 8, 50)))[0])
