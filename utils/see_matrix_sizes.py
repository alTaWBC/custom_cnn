import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import configparser
import json

config = configparser.ConfigParser()
config.read('../config.ini')

model = load_model(config.get('LOCATION', 'ModelLocation'))
input_shape = json.loads(config.get('MATRIX_SIZE', 'InitialSize'))

name = 'Initial Size'
size = input_shape
print('Initial Size', ' '*(25-len(name)), input_shape)
for i in range(len(model.layers)):
    layer = model.layers[i]
    name = layer.name

    if name.startswith('conv'):
        stride = layer.get_config()["strides"]
        size = conv_size(layer.get_weights()[0].shape, size, stride)
        
    if 'pool' in name:
        stride = layer.get_config()['pool_size']
        size = pool_size(size, stride)
        
    if name.startswith('dense'):
        size = (1, layer.get_weights()[0].shape[1])
        
    if name.startswith('dropout'):
        continue
    
    print(name, ' '*(25-len(name)), size)

def conv_size(filters, size, stride):
    dim1 = (size[1] - filters[0])//stride[0] + 1
    dim2 = (size[2] - filters[1])//stride[1] + 1
    
    return (1, dim1, dim2, filters[3])


def pool_size(size, stride):
    dim1 = size[1] - stride[0] + 1
    dim2 = size[2] - stride[1] + 1
    
    return (1, dim1, dim2, size[-1])
