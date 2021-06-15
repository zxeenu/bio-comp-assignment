import math
import sys

import matplotlib
import nnfs
import numpy as np
from nnfs.datasets import spiral_data
from nnfs.datasets import vertical_data
import matplotlib.pyplot as plt

# inputs = [
#     [1, 2, 3, 2.5],
#     [2.0, 5.0, -1.0, 2.0],
#     [-1.5, 2.7, 3.3, -0.8],
# ]


# weights = [
#     [0.2, 0.8, -0.5, 1.0],
#     [0.5, -0.91, 0.26, -0.5],
#     [-0.26, -0.27, 0.17, 0.87],
# ]
# biases = [2, 3, 0.5]

# weights2 = [
#     [0.1, -0.14, -0.5],
#     [-0.5, 0.12, -0.33],
#     [-0.44, 0.73, -0.13],
# ]
# biases2 = [1, -2, -0.5]

# layer1_output = np.dot(inputs, np.array(weights).T) + biases
# layer2_output = np.dot(layer1_output, np.array(weights2).T) + biases2
# print(layer2_output)

# inputs = [1, 2, 3, 2.5]
# biases = 2
# weights = [0.2, 0.8, -0.5, 1.0]
# output = np.dot(weights, inputs) + biases
# print(output)

# for neuron_weights, neuron_bias in zip(weights, biases):
#     neuron_output = 0
#     for n_input, weight in zip(inputs, neuron_weights):
#         neuron_output += n_input * weight
#     neuron_output += neuron_bias
#     layer_outputs.append(neuron_output)


nnfs.init()
np.random.seed(0)


# weights -iv 0.1 to +iv 0.1
# biases 0, or sometimes 1
class Layer_Dense:
    def __init__(self, n_inputs: int, n_neurons: int):
        """ 
        n_inputs has to be the number of elements the nested array will have, that should be passed in the self.forward method
        for example, if 3 is passed as n_inputs, a [[1, 2, 3], [1, 3, 5], [4, 3, 6]]. 
        """
        self.weights = 0.10 * np.random.randn(
            n_inputs, n_neurons
        )  # gausian distribution
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs: 'np.array'):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLu:
    def forward(self, inputs: 'np.array'):
        self.output = np.maximum(0, inputs)


class Activation_Softmax:
    def forward(self, inputs: 'np.array'):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

class Loss:
    def calculate(self, output: 'np.array', y: 'np.array'):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred: 'np.array', y_true: 'np.array'):
        """
        y_pred: the softmax output
        y_true: the class list / labels 
        """

        samples = len(y_pred)

        # in order to avoid intinities.
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7) 

        # means passed was a list of scalar values for the labels
        # this is a row by col indexing + the range iterates over the length, which is the size of the rows. 
        # The class should have as many elements, as there are rows in the softmax_output. 
        # Also, the softmax_output elements, each need to be the same dimen as the class
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

        # means passed was a one hot encoded list for the labels
        # if one hot encoded, when the y_pred multiplies against zero's, they are neurtalized
        # while the actual value is left.
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negetive_log_likelihoods = -np.log(correct_confidences)
        return negetive_log_likelihoods





# X_ = [
# [3.93124595e-02,  9.32828337e-03, 9.32828337e-03], 
# [3.93124595e-02,  9.32828337e-03, 9.32828337e-03], 
# [3.93124595e-02,  9.32828337e-03, 9.32828337e-03], 
# [3.93124595e-02,  9.32828337e-03, 9.32828337e-03]
# ]

# Y = np.array([1, 1, 1, 1])
# X = np.array(X_)
###########################

# X, Y = spiral_data(samples=100, classes=3)
X, Y = vertical_data(samples=100, classes=3)
 
# plt.scatter(X[:, 0], X[:, 1], c=Y, s=40, cmap='brg')
# plt.show()

# ----------------- #

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLu()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

loss_function = Loss_CategoricalCrossentropy()

lowest_loss = 999999
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

# ----------------- #

for iteration in range(1000):
    # update weights with some small random values
    dense1.weights += 0.05 * np.random.randn(2, 3)
    dense1.biases += 0.05 * np.random.randn(1, 3)
    dense2.weights += 0.05 * np.random.randn(3, 3)
    dense2.biases += 0.05 * np.random.randn(1, 3)

    # perform forward pass of our trainning data through this layer
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    # perform a forward pass through activation function
    # it takes the output of second dense layer here and returns loss
    loss = loss_function.calculate(activation2.output, Y)

    # calculate accuracy from output of activation2 and targets
    # calculate values along first axis
    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions == Y)
    print('loss: ', loss, 'acc: ', accuracy)

    # if loss is smaller - print and save weights and biases aside
    if loss < lowest_loss:
        print('New set of weights found, iteration: ', iteration, 'loss: ', loss, 'acc: ', accuracy)
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss
    else:
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()
        # print('Reverting to earlier weight set, iteration: ', iteration, 'loss: ', loss, 'acc: ', accuracy)


#################


# for iteration in range(1000):

    



# accuracy checking
# predictions = np.argmax(activation2.output, axis=1)
# accuracy = np.mean(predictions == Y)
# print(accuracy)
# print(predictions[1], "-", Y[1])






# print(activation2.output[:5])

# softmax_output = [0.7, 0.1, 0.2]
# target_output = [1, 0, 0]

# loss = -(
#             math.log(softmax_output[0]) * target_output[0] + 
#             math.log(softmax_output[1]) * target_output[1] + 
#             math.log(softmax_output[2]) * target_output[2] 
#         )

# print(loss)

# softmax_outputs = np.array([
#                             [0.7, 0.1, 0.2],
#                             [0.1, 0.5, 0.4],
#                             [0.02, 0.9, 0.08],
#                                             ])

# class_targets = [0, 1, 1]
# class_target_np = np.array(class_targets)
# # print(softmax_outputs[[0, 1, 2], [class_targets]])  # this is a row by col indexing
# print(softmax_outputs[range(len(softmax_outputs)), [class_targets]])  # this is a row by col indexing + the range iterates over the length, which is the size of the rows. The class should have as many elements, as there are rows in the softmax_output. Also, the softmax_output elements, each need to be the same dimen as the class
# # print(-np.log(softmax_outputs[[0, 1, 2], [class_targets]]))

# loss_f = Loss_CategoricalCrossentropy()

# v = loss_f.forward(softmax_outputs, class_target_np)
# print(v)

# inputs, how many features
# nurons, anything i want
# layer1 = Layer_Dense(n_inputs=4, n_neurons=5)
# layer2 = Layer_Dense(5, 2)

# layer1.forward(X)
# layer2.forward(layer1.output)
# print(layer2.output)


# X, Y = spiral_data(100, 3)
# layer1 = Layer_Dense(2, 5)
# activation1 = Activation_ReLu()

# layer1.forward(X)

# activation1.forward(layer1.output)
# # print(activation1.output)


# layer_outputs_u = [
#     [4.8, 1.21, 2.385],
#     [4.8, 1.21, 2.385],
#     [4.8, 1.21, 2.385],
# ]
# print(np.sum(layer_outputs_u, axis=1, keepdims=True), "--")

# we take every element in a row, exponent it with eulers number
# we take the sum of the row
# divide element by sum, to get normailized, non-negetive values
# exp_values = np.exp(layer_outputs_u)
# norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)
# print(norm_values)

# exp_values = []

# for output in layer_outputs_u:
#     exp_values.append(E ** output)

# print(exp_values)
# norm_base = sum(exp_values)
# norm_values = []

# for value in exp_values:
#     norm_values.append(value / norm_base)

# print(norm_values)
# print(sum(norm_values))
