import numpy as np

np.random.seed(0)

#  batch of inputs per layer
#  number of samples in batch can vary depending on problem size and dataset
X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

# create new layer object, used to create new layers.
# init lets us specify number of samples and number of neurons in the layer.
#
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # we use 0.1 to make it so that weights are between 1 and -1
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        # biases are technically only column vector hence this specific 'shape'
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

layer1 = Layer_Dense(4, 5)
layer2 = Layer_Dense(5, 2)

layer1.forward(X)
# print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)
