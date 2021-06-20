import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import configparser
import json

config = configparser.ConfigParser()
config.read('../config.ini')

model = load_model(config.get('LOCATION', 'ModelLocation'))

for layer in model.layers:
    name = layer.name
    
    if name.startswith('conv'):
        print(name, ' '*(25-len(name)), layer.get_config()["strides"])
    elif 'pool' in name:
        print(name, ' '*(25-len(name)), layer.get_config()['pool_size'])
