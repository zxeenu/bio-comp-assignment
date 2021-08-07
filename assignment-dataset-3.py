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
        self.inputs = inputs

    def backward(self, dvalues):
        # gradients of parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # gradients on values
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLu:
    def forward(self, inputs: 'np.array'):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

        # zero gradients where input values were negetive
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:
    def forward(self, inputs: 'np.array'):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues, y_true):
        # create unititialized array
        self.dinputs = np.empty_like(dvalues)

        # enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            #flatten output array
            single_output = single_output.reshape(-1, 1)

            #calculate jacobian matrix of the output and
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

            # calculate sample wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

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

    def backwards(self, dvalues, y_true):

        # umber of samples
        samples = len(dvalues)

        # number of labels in every sample
        # we'll use the first sample to count them
        labels = len(dvalues[0])

        # if labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y=true = np.eye(labels)[y_true]

        # calculate gradient
        self.dinputs = -y_true / dvalues

        # normalize gradient
        self.dinputs = self.dinputs / samples


class Activation_Softmax_Loss_CategoricalCrossentropy():

    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true):
        # output layers activation function
        self.activation.forward(inputs)

        #set the output
        self.output = self.activation.output

        # calcuate and return loss value
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        # number of samples
        samples = len(dvalues)

        # if labels are one-hot encoded
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # copy so we can work without mutating original values
        self.dinputs = dvalues.copy()

        # calculation of the gradient
        self.dinputs[range(samples), y_true] -= 1

        # normalize the calculate gradient
        self.dinputs = self.dinputs / samples

# X_ = [
# [3.93124595e-02,  9.32828337e-03, 9.32828337e-03], 
# [3.93124595e-02,  9.32828337e-03, 9.32828337e-03], 
# [3.93124595e-02,  9.32828337e-03, 9.32828337e-03], 
# [3.93124595e-02,  9.32828337e-03, 9.32828337e-03]
# ]

# Y = np.array([1, 1, 1, 1])
# X = np.array(X_)
###########################


 



def main():

    # X, Y = spiral_data(samples=100, classes=3)
    X, Y = vertical_data(samples=100, classes=3)

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

    plt.scatter(X[:, 0], X[:, 1], c=Y, s=40, cmap='brg')
    plt.show()

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


def main2():
    def f(x):
        return 2*x**2


    x = np.arange(0, 5, 0.001)
    y = f(x)

    plt.plot(x, y)


    # point and the close enough point
    p2_delta = 0.001
    x1 = 2
    x2 = x1 + p2_delta

    y1 = f(x1)
    y2 = f(x2)

    print((x1, y1), (x2, y2))

    # derivitive approximation and y-intercept for the tanget line
    approximate_derivative = (y2-y1) / (x2-x1)
    b = y2 - approximate_derivative*x2

    # We put the tangent line calculation into a function so we can call
    # it multiple times for different values of x
    # approximate_derivative and b are constant for given function
    # thus calculated once above this function
    def tangent_line(x):
        return approximate_derivative*x + b

    # plotting the tangent line
    # +/- 0.9 to draw the tangent line on our graph
    # then we calculate the y for given x using the tangent line function
    # Matplotlib will draw a line for us through these points

    to_plot = [x1-0.9, x1, x1+0.9]
    plt.plot(to_plot, [tangent_line(i) for i in to_plot])

    print('approximate derivitive for f(x)', f'where x = {x1} is {approximate_derivative}')


    # plt.plot(x, y)
    plt.show()


def main3():
    def f(x):
        return 2*x**2


    x = np.arange(0, 5, 0.001)
    y = f(x)

    plt.plot(x, y)

    colors = ['k', 'g', 'r', 'b', 'c']

    def approximate_tangent_line(x, approximate_derivitive):
        return (approximate_derivitive*x) + b


    for i in range(5):
        # point and the close enough point
        p2_delta = 0.001
        x1 = i
        x2 = x1 + p2_delta

        y1 = f(x1)
        y2 = f(x2)

        print((x1, y1), (x2, y2))

        # derivitive approximation and y-intercept for the tanget line
        approximate_derivative = (y2-y1) / (x2-x1)
        b = y2 - approximate_derivative*x2

        to_plot = [x1-0.9, x1, x1+0.9]
        plt.scatter(x1, y1, c=colors[i])

        plt.plot([point for point in to_plot], 
                 [approximate_tangent_line(point, approximate_derivative) for point in to_plot],
                 c=colors[i])

        print('approximate derivitive for f(x)', f'where x = {x1} is {approximate_derivative}')


        # plt.plot(x, y)
    plt.show()

def main4():
    x = [1.0, -2.0, 3.0] # input values
    w = [-3.0, -1.0, 2.0] # weights
    b = 1.0 # bias

    xw0 = x[0] * w[0]
    xw1 = x[1] * w[1]
    xw2 = x[2] * w[2]
    b = 1.0 # bias
    
    z = xw0 + xw1 + xw2 + b
    
    y = max(z, 0) # relu
    
    dvalue = 1.0 # an arbirary derivitive value from the next layer

    drelu_dz = dvalue * (1.0 if z > 0 else 0.0) # derivative of Relu and the chain rule
    print(drelu_dz)

    # partial derivatives of the multiplication, the chain rule
    dsum_dxw0 = 1
    dsum_dxw1 = 1
    dsum_dxw2 = 1
    dsum_db = 1
    drelu_dxw0 = drelu_dz * dsum_dxw0
    drelu_dxw1 = drelu_dz * dsum_dxw1
    drelu_dxw2 = drelu_dz * dsum_dxw2
    drelu_db = drelu_dz * dsum_db
    print(drelu_dxw0, drelu_dxw1, drelu_dxw2, drelu_db)

    # # prtial derivatives of the multiplication, the chain rule
    dmul_dx0 = w[0]
    dmul_dx1 = w[1]
    dmul_dx2 = w[2]
    dmul_dw0 = x[0]
    dmul_dw1 = x[1]
    dmul_dw2 = x[2]
    drelu_dx0 = drelu_dxw0 * dmul_dx0
    drelu_dw0 = drelu_dxw0 * dmul_dw0
    drelu_dx1 = drelu_dxw1 * dmul_dx1
    drelu_dw1 = drelu_dxw1 * dmul_dw1
    drelu_dx2 = drelu_dxw2 * dmul_dx2
    drelu_dw2 = drelu_dxw2 * dmul_dw2
    print(drelu_dx0, drelu_dw0, drelu_dx1, drelu_dw1, drelu_dx2, drelu_dw2)


def main5():
    dvalues = np.array([[1., 1., 1.]])

    weights = np.array([[0.2, 0.8, -0.5, 1],
                        [0.5, -0.91, 0.26, -0.5],
                        [-0.26, -0.27, 0.17, 0.87]]).T


    dx0 = sum(weights[0]*dvalues[0])
    dx1 = sum(weights[1]*dvalues[0])
    dx2 = sum(weights[2]*dvalues[0])
    dx3 = sum(weights[3]*dvalues[0])

    dinputs = np.array([dx0, dx1, dx2, dx3])
    print(dinputs)

    dinputs_ = np.dot(dvalues[0], weights.T)
    print(dinputs_)

    print("-------")
    dvalues_ = np.array([[1., 1., 1.],
                        [2., 2., 2.],
                        [3., 3., 3.]])
    dinputs_2 = np.dot(dvalues_, weights.T)
    print(dinputs_2)


if __name__ == "__main__":
    # main()
    # main3()
    main5()


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
