import cnn_single_functions as helper
import numpy as np
from log_mel import log_mel
import time

conv1 = r"C:\Users\WilliamCosta\Desktop\repositories\updated_tensorflow\weights\vocal_folds\0_conv2d_1_kernel_0.npy"
conv2 = r"C:\Users\WilliamCosta\Desktop\repositories\updated_tensorflow\weights\vocal_folds\4_conv2d_2_kernel_0.npy"
den1 = r"C:\Users\WilliamCosta\Desktop\repositories\updated_tensorflow\weights\vocal_folds\8_dense_1_kernel_0.npy"
den2 = r"C:\Users\WilliamCosta\Desktop\repositories\updated_tensorflow\weights\vocal_folds\10_dense_2_kernel_0.npy"
den3 = r"C:\Users\WilliamCosta\Desktop\repositories\updated_tensorflow\weights\vocal_folds\12_dense_3_kernel_0.npy"
den4 = r"C:\Users\WilliamCosta\Desktop\repositories\updated_tensorflow\weights\vocal_folds\14_dense_4_kernel_0.npy"
den5 = r"C:\Users\WilliamCosta\Desktop\repositories\updated_tensorflow\weights\vocal_folds\16_dense_5_kernel_0.npy"

weights1 = np.load(conv1)
weights2 = np.load(conv2)
weights3 = np.load(den1)
weights4 = np.load(den2)
weights5 = np.load(den3)
weights6 = np.load(den4)
weights7 = np.load(den5)


def convolutions_first_layer(matrix, weights, stride):
    final_array = []

    for i in range(weights.shape[3]):
        weight = weights[:, :, 0, i]
        convolution = helper.convolution(matrix, weight, stride)
        final_array.append(convolution)
    return np.array(final_array)


def convolutions_second_layer(matrix, weights, stride):
    kernel_transposed = np.transpose(weights2, (3, 2, 0, 1))
    final_array = []

    for i in range(kernel_transposed.shape[0]):
        weight = kernel_transposed[i, :, :, :]
        convolution = helper.convolution2(matrix, weight, stride)
        final_array.append(convolution[0])
    return np.array(final_array)


def batch_normalizations(matrix, batch_size):
    final_array = np.ones(matrix.shape)
    for i in range(0, matrix.shape[0], batch_size):
        start, end = i, min(i-1 + batch_size, matrix.shape[0] - 1)
        batch = matrix[start:end, :, :]
        final_array[start:end, :, :] = helper.batch_normalization(batch)
    return final_array


def activations(matrix):
    return helper.activation(matrix)


def max_norm(matrix):
    norms = np.sqrt(np.sum(np.square(matrix)))
    desired = np.clip(norms, 0, 2)
    return matrix * (desired / (1e-07 + norms))


def softmax(matrix):
    return helper.softmax(matrix)


def poolings(matrix, poolWindow):
    final_array = []
    for i in range(matrix.shape[0]):
        final_array.append(helper.pooling(matrix[i, :, :], poolWindow))
    return np.array(final_array)


def dense(matrix, weights):
    final_array = []
    for i in range(weights.shape[1]):
        final_array.append(helper.dense(matrix, weights[:, i]))
    return np.array(final_array)


def run_sample(matrix):
    lista = [time.time()]
    lista2 = []
    convolution1 = convolutions_first_layer(matrix, max_norm(weights1), (2, 1))
    lista2.append("convolution1")
    lista.append(time.time())
    print(convolution1.shape)
    batch1 = batch_normalizations(convolution1, 10)
    lista2.append("batch1")
    lista.append(time.time())
    print(batch1.shape)
    activation1 = activations(batch1)
    lista2.append("activation1")
    lista.append(time.time())
    print(activation1.shape)
    pooling1 = poolings(activation1, (2, 2))
    lista2.append("pooling1")
    lista.append(time.time())
    print(pooling1.shape)
    convolution2 = convolutions_second_layer(
        pooling1, max_norm(weights2), (1, 1, 1))
    lista2.append("convolution2")
    lista.append(time.time())
    print(convolution2.shape)
    batch2 = batch_normalizations(convolution2, 10)
    lista2.append("batch2")
    lista.append(time.time())
    print(batch2.shape)
    activation2 = activations(batch2)
    lista2.append("activation2")
    lista.append(time.time())
    print(activation2.shape)
    pooling2 = poolings(activation2, (2, 2))
    lista2.append("pooling2")
    lista.append(time.time())
    print(pooling2.shape)
    flatten = pooling2.flatten()
    lista.append(time.time())
    print(flatten.shape)
    dense1 = activations(dense(flatten, max_norm(weights3)))
    lista2.append("dense1")
    lista.append(time.time())
    print(dense1.shape)
    dense2 = activations(dense(dense1, max_norm(weights4)))
    lista2.append("dense2")
    lista.append(time.time())
    print(dense2.shape)
    dense3 = activations(dense(dense2, max_norm(weights5)))
    lista2.append("dense3")
    lista.append(time.time())
    print(dense3.shape)
    dense4 = activations(dense(dense3, max_norm(weights6)))
    lista2.append("dense4")
    lista.append(time.time())
    print(dense4.shape)
    dense5 = dense(dense4, weights7)
    lista2.append("dense5")
    lista.append(time.time())
    print(dense5.shape)
    print(dense5)
    difference = [lista[i+ 1] - lista[i] for i in range(len(lista)- 1) ]
    
    for name, times in zip(lista2, difference):
        print(name, times)
    
    return softmax(dense5)



def classify(file_loc):
    return run_sample(log_mel(file_loc))


start = time.time()
# classify(r"C:\Users\WilliamCosta\Desktop\repositories\updated_tensorflow\001_CH_L_R.wav")
log_mel = log_mel(r"C:\Users\WilliamCosta\Desktop\repositories\updated_tensorflow\001_CH_L_R.wav")
end = time.time()
run_sample(log_mel)
end_ = time.time()
print(end - start, end_ - end)
