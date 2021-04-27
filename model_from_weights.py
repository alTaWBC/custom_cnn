from tensorflow import keras
from model import first_model
import os
import numpy as np


def sort_number(word):
    number = int(word.split('_')[0])
    if number > 19:
        number = float(f"7.{number-19}")
    elif number > 17:
        number = float(f"3.{number-17}")
    return number
    # a = int(word1.split('_')[0])
    # b = int(word2.split('_')[0])
    # return (a > b) - (a < b)


# %%% Vocal Folds
responses = 2
model = first_model(responses)
vocal_folds_location = r"C:\Users\WilliamCosta\Desktop\repositories\updated_tensorflow\weights\vocal_folds"

filenames = sorted([f for f in os.listdir(vocal_folds_location)
                   if f.endswith('.npy')], key=sort_number)
print(filenames)

weights = []

for f in filenames:
    number = float(f.split('_')[0])
    weights.append(np.load(os.path.join(vocal_folds_location, f)))


# #%%% Place of Articulation
# responses = 3
# model = first_model(responses)
# place_articulation_location = r"C:\Users\WilliamCosta\Desktop\repositories\updated_tensorflow\weights\place_articulation"

# for layer in model.weights:
#     print(layer.name)

# for weight in model.weights:
#     print(weight.shape)

print(len(weights), len(model.weights))

for weight, i in zip(model.weights, weights):
    print(f"{weight.name}: {weight.shape} --> {i.shape}")
model.set_weights(weights)

newModelLocation = r"C:\Users\WilliamCosta\Desktop\repositories\updated_tensorflow\updated_model"
model.save(os.path.join(newModelLocation, "vocal_folds"))


#%%%
modelLocation = r"C:\Users\WilliamCosta\Desktop\repositories\updated_tensorflow\updated_model\vocal_folds"

model = keras.models.load_model(modelLocation)
model.summary()
model.predict(np.zeros((1, 80, 9, 1)))

# %%
