import convolution
import numpy as np


convolution.profile(3)


print(convolution.predict_layer(np.zeros((1, 80, 9, 1)), -1))

print()
print(convolution.predict_cnn(np.zeros((1, 80, 9, 1), dtype=np.float32))[0])
