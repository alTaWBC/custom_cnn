import numpy as np
from keras.engine.saving import load_model
from keras.models import Model
import tensorflow as tf
import json
import time
import configparser
cimport numpy as np
import cython
DTYPE = np.float32

ctypedef np.float32_t DTYPE_t

config = configparser.ConfigParser()
config.read('./config.ini')
weights_location = config['LOCATION']['weightslocation']
model_location = config['LOCATION']['modellocation']
separator = config['LOCATION']['separator']

cdef float[:,:,:,::1] c1_weights= np.load(f"{weights_location}{separator}{config['NAMES']['firstconvolution']}.npy").astype(DTYPE),
cdef float[::1] c1_biases= np.load(f"{weights_location}{separator}{config['NAMES']['firstconvolutionbias']}.npy").astype(DTYPE),

cdef float b1_epsilon= np.load(f"{weights_location}{separator}{config['NAMES']['firstbatchepsilon']}.npy").astype(DTYPE)[0],
cdef float[::1] b1_beta = np.load(f"{weights_location}{separator}{config['NAMES']['firstbatchbeta']}.npy").astype(DTYPE),
cdef float[::1] b1_gamma= np.load(f"{weights_location}{separator}{config['NAMES']['firstbatchgamma']}.npy").astype(DTYPE),
cdef float[::1] b1_moving_mean= np.load(f"{weights_location}{separator}{config['NAMES']['firstbatchmovingmean']}.npy").astype(DTYPE),
cdef float[::1] b1_moving_var= np.load(f"{weights_location}{separator}{config['NAMES']['firstbatchmovingvar']}.npy").astype(DTYPE),

cdef float[:,:,:,::1] c2_weights= np.load(f"{weights_location}{separator}{config['NAMES']['secondconvolution']}.npy").astype(DTYPE),
cdef float[::1] c2_biases= np.load(f"{weights_location}{separator}{config['NAMES']['secondconvolutionbias']}.npy").astype(DTYPE),

cdef float b2_epsilon= np.load(f"{weights_location}{separator}{config['NAMES']['secondbatchepsilon']}.npy").astype(DTYPE)[0],
cdef float[::1] b2_beta= np.load(f"{weights_location}{separator}{config['NAMES']['secondbatchbeta']}.npy").astype(DTYPE),
cdef float[::1] b2_gamma= np.load(f"{weights_location}{separator}{config['NAMES']['secondbatchgamma']}.npy").astype(DTYPE),
cdef float[::1] b2_moving_mean= np.load(f"{weights_location}{separator}{config['NAMES']['secondbatchmovingmean']}.npy").astype(DTYPE),
cdef float[::1] b2_moving_var= np.load(f"{weights_location}{separator}{config['NAMES']['secondbatchmovingvar']}.npy").astype(DTYPE),

cdef float[:,:] d1_weights= np.load(f"{weights_location}{separator}{config['NAMES']['firstdense']}.npy").astype(DTYPE).T,
cdef float[::1] d1_biases= np.load(f"{weights_location}{separator}{config['NAMES']['firstdensebias']}.npy").astype(DTYPE),

cdef float[:,:] d2_weights= np.load(f"{weights_location}{separator}{config['NAMES']['seconddense']}.npy").astype(DTYPE).T,
cdef float[::1] d2_biases= np.load(f"{weights_location}{separator}{config['NAMES']['seconddensebias']}.npy").astype(DTYPE),

cdef float[:,:] d3_weights= np.load(f"{weights_location}{separator}{config['NAMES']['thirddense']}.npy").astype(DTYPE).T,
cdef float[::1] d3_biases= np.load(f"{weights_location}{separator}{config['NAMES']['thirddensebias']}.npy").astype(DTYPE),

cdef float[:,:] d4_weights= np.load(f"{weights_location}{separator}{config['NAMES']['fourthdense']}.npy").astype(DTYPE).T,
cdef float[::1] d4_biases= np.load(f"{weights_location}{separator}{config['NAMES']['fourthdensebias']}.npy").astype(DTYPE),

cdef float[:,:] d5_weights= np.load(f"{weights_location}{separator}{config['NAMES']['fifthdense']}.npy").astype(DTYPE).T,
cdef float[::1] d5_biases= np.load(f"{weights_location}{separator}{config['NAMES']['fifthdensebias']}.npy").astype(DTYPE),

# Aos pesos foram aplicados o max norm
# Variaveis são definidas no inicio

model = load_model(model_location)

model_input = model.input
models = []
layers = {}


def unpack_layer(layer):
    if any([True if name in layer.name else False for name in ['pool', 'activation', 'flatten', 'dropout']]):
        return
    elif 'batch' in layer.name:
        gamma, beta, mean, var = layer.get_weights()
        layers[layer.name] = {
            'moving_mean': mean.astype(DTYPE),
            'moving_var': var.astype(DTYPE),
            "epsilon": layer.epsilon,
            'gamma': gamma.astype(DTYPE),
            'beta': beta.astype(DTYPE),
        }
    elif 'conv' in layer.name or 'dense' in layer.name:
        layers[layer.name] = {
            'weights': layer.get_weights()[0].astype(DTYPE),
            'biases': layer.get_weights()[1].astype(DTYPE)
        }


for layer in model.layers:
    layer_model = Model(inputs=model_input, outputs=layer.output)
    models.append(layer_model)
    unpack_layer(layer)


def predict(array, indices=range(len(models))):
    for i in indices:
        print(models[i].predict(array))


def differences(matrix1, matrix2):
    dif = np.abs(matrix1 - matrix2)
    print("Mean +/- Std:", np.mean(dif), '+/-', np.std(dif))
    print("Varies Between:", np.min(dif), np.max(dif))


def predict_layer(array, layer):
    return models[layer].predict(array)


def max_norm(matrix):
    return np.minimum(np.maximum(matrix, -2), 2)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[DTYPE_t, ndim=4] convolution_2D(np.ndarray[DTYPE_t, ndim=4] matrix, stride):
    global c1_weights, c1_biases
    cdef np.ndarray[DTYPE_t, ndim=4] convolution = np.empty(json.loads(config['MATRIX_SIZE']['FirstConvolution']), dtype=DTYPE)
    cdef float [:,:,:,:] convolution_mv = convolution
    cdef int i,j,k

    for i in range(convolution.shape[1]):
        for j in range(convolution.shape[2]):
            for k in range(convolution.shape[3]):
                convolution_mv[0, i, j, k] = np.sum(
                    matrix[0, i*stride[0]:i*stride[0]+c1_weights.shape[0], j *
                           stride[1]:j*stride[1]+c1_weights.shape[1], 0] * c1_weights[:, :, 0, k]
                ) + c1_biases[k]
    return convolution

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[DTYPE_t, ndim=4] convolution_2D_2(np.ndarray[DTYPE_t, ndim=4] matrix, stride):
    global c2_weights, c2_biases
    cdef np.ndarray[DTYPE_t, ndim=4] convolution = np.empty(json.loads(config['MATRIX_SIZE']['SecondConvolution']), dtype=DTYPE)
    cdef float [:,:,:,:] convolution_mv = convolution
    cdef int i,j,k

    for i in range(convolution.shape[1]):
        for j in range(convolution.shape[2]):
            for k in range(convolution.shape[3]):
                convolution_mv[0, i, j, k] = np.sum(
                    matrix[0, i*stride[0]:i*stride[0]+c2_weights.shape[0], j *
                           stride[1]:j*stride[1]+c2_weights.shape[1], :] * c2_weights[:, :, :, k]
                ) + c2_biases[k]
    return convolution

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[DTYPE_t, ndim=4] batch_normalization(np.ndarray[DTYPE_t, ndim=4] matrix):
    global b1_moving_mean, b1_moving_var, b1_epsilon, b1_gamma, b1_beta
    cdef np.ndarray[DTYPE_t, ndim=4] batch_normalization = np.empty(json.loads(config['MATRIX_SIZE']['FirstBatch']), dtype=DTYPE)
    cdef float [:,:,:,:] batch_normalization_mv = batch_normalization
    cdef int i,j,k

    for i in range(batch_normalization.shape[1]):
        for j in range(batch_normalization.shape[2]):
            for k in range(batch_normalization.shape[3]):
                batch_normalization_mv[0, i, j, k] = (matrix[0, i, j, k] - b1_moving_mean[k]) / \
                    np.sqrt(b1_moving_var[k] + b1_epsilon) * b1_gamma[k] + b1_beta[k]

    return batch_normalization

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[DTYPE_t, ndim=4] batch_normalization_2(matrix):
    global b2_moving_mean, b2_moving_var, b2_epsilon, b2_gamma, b2_beta
    cdef np.ndarray[DTYPE_t, ndim=4] batch_normalization = np.empty(json.loads(config['MATRIX_SIZE']['SecondBatch']), dtype=DTYPE)
    cdef float [:,:,:,:] batch_normalization_mv = batch_normalization
    cdef int i,j,k

    for i in range(batch_normalization.shape[1]):
        for j in range(batch_normalization.shape[2]):
            for k in range(batch_normalization.shape[3]):
                batch_normalization_mv[0, i, j, k] = (matrix[0, i, j, k] - b2_moving_mean[k]) / \
                    np.sqrt(b2_moving_var[k] + b2_epsilon) * b2_gamma[k] + b2_beta[k]

    return batch_normalization


def activation(matrix):
    return np.maximum(matrix, 0)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[DTYPE_t, ndim=4] pooling(np.ndarray[DTYPE_t, ndim=4] matrix, stride):
    cdef np.ndarray[DTYPE_t, ndim=4] pool = np.empty(json.loads(config['MATRIX_SIZE']['FirstPool']), dtype=DTYPE)
    cdef float [:,:,:,:] pool_mv = pool
    cdef int i,j,k

    for i in range(pool.shape[1]):
        for j in range(pool.shape[2]):
            for k in range(pool.shape[3]):
                pool_mv[0, i, j, k] = np.max(
                    matrix[0, i:i+stride[0], j:j+stride[1], k])
    return pool

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[DTYPE_t, ndim=4] pooling_2(np.ndarray[DTYPE_t, ndim=4] matrix, stride):
    cdef np.ndarray[DTYPE_t, ndim=4] pool = np.empty(json.loads(config['MATRIX_SIZE']['SecondPool']), dtype=DTYPE)
    cdef float [:,:,:,:] pool_mv = pool
    cdef int i,j,k

    for i in range(pool.shape[1]):
        for j in range(pool.shape[2]):
            for k in range(pool.shape[3]):
                pool_mv[0, i, j, k] = np.max(
                    matrix[0, i:i+stride[0], j:j+stride[1], k])
    return pool

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[DTYPE_t, ndim=2] dense1(np.ndarray[DTYPE_t, ndim=1] matrix):
    global d1_weights, d1_biases
    cdef np.ndarray[DTYPE_t, ndim=2] dense = np.empty(json.loads(config['MATRIX_SIZE']['FirstDense']), dtype=DTYPE)
    cdef float [:, :] dense_mv = dense
    cdef int i

    for i in range(dense.shape[1]):
        dense_mv[0, i] = activation(np.dot(matrix, d1_weights[i]) + d1_biases[i])
    return dense

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[DTYPE_t, ndim=2] dense2(np.ndarray[DTYPE_t, ndim=2] matrix):
    global d2_weights, d2_biases
    cdef np.ndarray[DTYPE_t, ndim=2] dense = np.empty(json.loads(config['MATRIX_SIZE']['SecondDense']), dtype=DTYPE)
    cdef float [:, :] dense_mv = dense
    cdef int i

    for i in range(dense.shape[1]):
        dense_mv[0, i] = activation(np.dot(matrix, d2_weights[i]) + d2_biases[i])
    return dense

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[DTYPE_t, ndim=2] dense3(np.ndarray[DTYPE_t, ndim=2] matrix):
    global d3_weights, d3_biases
    cdef np.ndarray[DTYPE_t, ndim=2] dense = np.empty(json.loads(config['MATRIX_SIZE']['ThirdDense']), dtype=DTYPE)
    cdef float [:, :] dense_mv = dense
    cdef int i

    for i in range(dense.shape[1]):
        dense_mv[0, i] = activation(np.dot(matrix, d3_weights[i]) + d3_biases[i])
    return dense

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[DTYPE_t, ndim=2] dense4(np.ndarray[DTYPE_t, ndim=2] matrix):
    global d4_weights, d4_biases
    cdef np.ndarray[DTYPE_t, ndim=2] dense = np.empty(json.loads(config['MATRIX_SIZE']['FourthDense']), dtype=DTYPE)
    cdef float [:, :] dense_mv = dense
    cdef int i

    for i in range(dense.shape[1]):
        dense_mv[0, i] = activation(np.dot(matrix, d4_weights[i]) + d4_biases[i])
    return dense

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[DTYPE_t, ndim=2] dense5(np.ndarray[DTYPE_t, ndim=2] matrix):
    global d5_weights, d5_biases
    cdef np.ndarray[DTYPE_t, ndim=2] dense = np.empty(json.loads(config['MATRIX_SIZE']['FifthDense']), dtype=DTYPE)
    cdef float [:, :] dense_mv = dense
    cdef np.ndarray[DTYPE_t, ndim=2] exp_matrix
    cdef int i

    for i in range(dense.shape[1]):
        dense_mv[0, i] = activation(np.dot(matrix, d5_weights[i]) + d5_biases[i])

    exp_matrix = np.exp(dense)
    return exp_matrix / np.sum(exp_matrix)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef predict_cnn(matrix):
    cdef np.ndarray[DTYPE_t, ndim=4] conv, batch, pool1, pool2
    cdef np.ndarray[DTYPE_t, ndim=2] dense
    cdef np.ndarray[DTYPE_t, ndim=1] flatten
    cdef np.ndarray acti

    durations = [time.time()]
    names = []

    conv = convolution_2D(matrix, (2, 1))
    durations.append(time.time())
    names.append("conv")
    batch = batch_normalization(conv)
    durations.append(time.time())
    names.append("batch")
    acti = activation(batch)
    durations.append(time.time())
    names.append("acti")
    pool1 = pooling(acti, (2, 2))
    durations.append(time.time())
    names.append("pool1")
    conv = convolution_2D_2(pool1, (1, 1, 1))
    durations.append(time.time())
    names.append("conv")
    batch = batch_normalization_2(conv)
    durations.append(time.time())
    names.append("batch")
    acti = activation(batch)
    durations.append(time.time())
    names.append("acti")
    pool2 = pooling_2(acti, (2, 2))
    durations.append(time.time())
    names.append("pool2")
    flatten = pool2.flatten()
    durations.append(time.time())
    names.append("flatten")
    dense = dense1(flatten)
    durations.append(time.time())
    names.append("dense")
    dense = dense2(dense)
    durations.append(time.time())
    names.append("dense")
    dense = dense3(dense)
    durations.append(time.time())
    names.append("dense")
    dense = dense4(dense)
    durations.append(time.time())
    names.append("dense")
    dense = dense5(dense)
    durations.append(time.time())
    names.append("dense")

    differences = [durations[i+1] - durations[i] for i in range(len(names))]

    # for name, duration in zip(names, differences):
    #     print(name, duration)

    return dense, differences


def profile(iterations, output_file='results.txt'):
    record = []
    names = ["conv1", "batch1", "activation1", "pooling1", "conv2", "batch2", "activation2",
             "pooling2", "flatten", "dense1", "dense2", "dense3", "dense4", "dense5", ]
    for i in range(iterations):
        random_matrix = np.random.rand(*json.loads(config['MATRIX_SIZE']['initialsize'])).astype(DTYPE)
        (_, durations) = predict_cnn(random_matrix)
        record.append(durations)

    array = np.array(record)
    mean = np.mean(array, axis=0)
    std = np.std(array, axis=0)
  
    try:
        with open(output_file, 'w') as outfile:
            print('Opened File')
            for i in range(array.shape[1]):
                outfile.write(f"{names[i]}: {mean[i]*1000:0.2f}, {std[i]*1000:0.2f}\n")
    except:

        print(f'File or folder {output_file} does not exist.\nPrinting results in console')
    finally:
        for i in range(array.shape[1]):
                print(f"{names[i]}: {mean[i]*1000:0.2f}, {std[i]*1000:0.2f}")
