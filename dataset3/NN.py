import matplotlib.pyplot as plt
import nnfs
import numpy as np
from nnfs.datasets import spiral_data, vertical_data


class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs

    def backward(self, delta_values):
        self.delta_weights = np.dot(self.inputs.T, delta_values)
        self.delta_biases = np.sum(delta_values, axis=0, keepdims=True)
        self.delta_inputs = np.dot(delta_values, self.weights.T)


class ActivationLeakyRelu:
    def forward(self, inputs):
        self.inputs = inputs  # save in case needed later
        self.output = np.maximum(0.1 * inputs, inputs)

    def backward(self, delta_values):
        self.delta_inputs = delta_values.copy()  # save copy for later


class ActivationRelu:
    def forward(self, inputs):
        self.inputs = inputs  # save in case needed later
        self.output = np.maximum(0, inputs)

    def backward(self, delta_values):
        self.delta_inputs = delta_values.copy()  # save copy for later
        self.delta_inputs[self.inputs <= 0] = 0  # trim off the negetive values


class ActivationSoftmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, delta_values):
        self.delta_inputs = np.empty_like(delta_values)

        for count, (single_output, single_delta_output) in enumerate(
            zip(self.output, delta_values)
        ):

            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(
                single_output, single_output.T
            )

            self.delta_inputs[count] = np.dot(jacobian_matrix, single_output)


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

    def backward(self, delta_values, y_true):
        samples_size = len(delta_values)
        labels = len(delta_values[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.delta_inputs = -y_true / delta_values
        self.delta_inputs = self.delta_inputs / samples_size

    def forward(self, y_pred, y_true):
        samples_size = len(y_pred)
        value_close_to_zero_but_not = 1e-7
        y_pred_clipped = np.clip(
            y_pred, value_close_to_zero_but_not, 1 - value_close_to_zero_but_not
        )

        # taking one hot encoding and matrix encoding into account
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples_size), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negetive_log_likeliness = -np.log(correct_confidences)
        return negetive_log_likeliness


class ActivationSoftmaxLoss:
    def __init__(self):
        self.softmax_activation = ActivationSoftmax()
        self.loss_function = Loss()

    def forward(self, inputs, y_true):
        self.softmax_activation.forward(inputs)
        self.output = self.softmax_activation.output
        return self.loss_function.calculate(self.output, y_true)

    def backward(self, delta_values, y_true):
        samples_size = len(delta_values)

        # account for multple types of target encoding
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.delta_inputs = delta_values.copy()
        self.delta_inputs[range(samples_size), y_true] -= 1  # calculate nudge values
        self.delta_inputs = self.delta_inputs / samples_size


class Optimizer:
    def __init__(self, learning_rate=0.5):
        self.learning_rate = learning_rate

    def update_weights_biases(self, layer):
        layer.weights += -self.learning_rate * layer.delta_weights
        layer.biases += -self.learning_rate * layer.delta_biases


class NeuralNetwork:
    def __init__(
        self, _iterations, _trainning_X, _trainning_Y, _test_Y, _learning_rate
    ):
        self.iterations = _iterations
        self.trainning_X = _trainning_X
        self.trainning_Y = _trainning_Y
        self.learning_rate = _learning_rate
        self.test_Y = _test_Y

        self.accuracy_records_trainning = []
        self.loss_records_trainning = []

        self.accuracy_records_test = []
        self.loss_records_test = []

    def train(self):
        self.dense1 = Layer(2, 64)
        self.actvation1 = ActivationRelu()
        self.dense2 = Layer(64, 3)
        self.optimizer = Optimizer(self.learning_rate)
        self.loss_and_activation_func = ActivationSoftmaxLoss()

        count = 0
        for iteration_number in range(self.iterations):
            # forward pass
            self.dense1.forward(self.trainning_X)
            self.actvation1.forward(self.dense1.output)
            self.dense2.forward(self.actvation1.output)
            loss = self.loss_and_activation_func.forward(
                self.dense2.output, self.trainning_Y
            )

            predictions = np.argmax(self.loss_and_activation_func.output, axis=1)
            if len(self.trainning_Y.shape) == 2:
                self.trainning_Y = np.argmax(self.trainning_Y, axis=1)
            accuracy = np.mean(predictions == self.trainning_Y)

            # for graphing
            self.accuracy_records_trainning.append(accuracy)
            self.loss_records_trainning.append(loss)

            # backward pass
            self.loss_and_activation_func.backward(
                self.loss_and_activation_func.output, self.trainning_Y
            )
            self.dense2.backward(self.loss_and_activation_func.delta_inputs)
            self.actvation1.backward(self.dense2.delta_inputs)
            self.dense1.backward(self.actvation1.delta_inputs)

            # update the weights and their respective baises
            self.optimizer.update_weights_biases(self.dense1)
            self.optimizer.update_weights_biases(self.dense2)

            count += 1
            if count > 100:
                print(
                    "Mode: Trainning",
                    "| " "Iteration: ",
                    iteration_number,
                    "| " "Loss: ",
                    loss,
                    "| " "Accuracy: ",
                    accuracy,
                )
                count = 0
        print("------------------")

    def test(self):
        count = 0
        for iteration_number in range(self.iterations):
            # forward pass
            self.dense1.forward(self.trainning_X)
            self.actvation1.forward(self.dense1.output)
            self.dense2.forward(self.actvation1.output)
            loss = self.loss_and_activation_func.forward(
                self.dense2.output, self.trainning_Y
            )

            predictions = np.argmax(self.loss_and_activation_func.output, axis=1)
            if len(self.trainning_Y.shape) == 2:
                self.trainning_Y = np.argmax(self.trainning_Y, axis=1)
            accuracy = np.mean(predictions == self.trainning_Y)

            # for graphing
            self.accuracy_records_test.append(accuracy)
            self.loss_records_test.append(loss)

            # backward pass
            self.loss_and_activation_func.backward(
                self.loss_and_activation_func.output, self.trainning_Y
            )
            self.dense2.backward(self.loss_and_activation_func.delta_inputs)
            self.actvation1.backward(self.dense2.delta_inputs)
            self.dense1.backward(self.actvation1.delta_inputs)

            count += 1
            if count > 100:
                print(
                    "Mode: Test",
                    "| " "Iteration: ",
                    iteration_number,
                    "| " "Loss: ",
                    loss,
                    "| " "Accuracy: ",
                    accuracy,
                )
                count = 0
        print("------------------")

    def graph(self):
        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle("DATASET 3")
        plt.subplots_adjust(
            left=0.125, bottom=0.1, right=0.9, top=0.880, wspace=0.2, hspace=0.585
        )
        x_axis = [x for x in range(0, self.iterations)]

        ###
        # ax1.set_title("Accuracy")
        ax1.grid()
        ax1.plot(x_axis, self.accuracy_records_trainning, color="g", label="Trainning")
        ax1.plot(x_axis, self.accuracy_records_test, color="r", label="Test")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Accuracy")
        ax1.legend(prop={"size": 6}, loc="lower right")
        ###
        # ax2.set_title("Loss")
        ax2.grid()
        ax2.plot(x_axis, self.loss_records_trainning, color="g", label="Trainning")
        ax2.plot(x_axis, self.loss_records_test, color="r", label="Test")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Loss")
        ax2.legend(prop={"size": 6}, loc="lower right")
        plt.show()


nnfs.init()
X, y = spiral_data(200, 3)

n = NeuralNetwork(1000, X, y, y, 0.5)
n.train()
n.test()
n.graph()
