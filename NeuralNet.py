import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

np.random.seed(0)


# create new layer object, used to create new layers.
# init lets us specify number of samples and number of neurons in the layer.
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # we use 0.1 to make it so that weights are between 1 and -1
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        # biases are technically only column vectors hence this specific 'shape'
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


# x and y are just coordinates and we get this from the spiral data set

X, y = spiral_data(samples=100, classes=3)

# at first we have a batch of two inputs which get put into 3 neurons
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

# since there are 3 outputs in layer 1, there have to be 3 in
dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])
