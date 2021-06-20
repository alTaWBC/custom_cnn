from keras.engine.saving import load_model
from os.path import join
from numpy import save, array, savetxt
import configparser
import json

config = configparser.ConfigParser()
config.read('../config.ini')

MODEL_LOCATION = config.get('LOCATION', 'ModelLocation')

model = load_model(MODEL_LOCATION)


def unpack_layer(layer):
    if any([True if name in layer.name else False for name in ['pool', 'activation', 'flatten', 'dropout']]):
        return
    elif 'batch' in layer.name:
        gamma, beta, mean, var = layer.get_weights()
        return {
            'moving_mean': mean,
            'moving_var': var,
            "epsilon": layer.epsilon,
            'gamma': gamma,
            'beta': beta,
        }
    elif 'conv' in layer.name or 'dense' in layer.name:
        return {
            'weights': layer.get_weights()[0],
            'biases': layer.get_weights()[1]
        }


def write_shapes(file, shapes):
    with open(file, 'w') as shape_file:
        for key, value in shapes.items():
            shape_file.write(f"{key}={value}\n")


reshape_values = {}
for layer in model.layers:
    name = layer.name
    layer_variables = unpack_layer(layer)
    if layer_variables is None:
        continue

    for key, value in layer_variables.items():
        if key is 'epsilon':
            value = array([value])
        save(join('..\\weights', f"{name}_{key}.npy"), value)
        reshape_values[f"{name}_{key}"] = value.shape

write_shapes(join('..\\weights','weights_shapes.txt'), reshape_values)
