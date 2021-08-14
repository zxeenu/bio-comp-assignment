import numpy as np
import pandas as pd

from neuralnetwork import NeuralNetwork

############# DATA SET 3 LOGIC START #############
dataset3 = pd.read_csv("data3.txt", delim_whitespace=True, dtype=str)
dataset3_len = len(dataset3)
dataset3_prepared_X = []
dataset3_prepared_Y = []

for num in range(0, dataset3_len):
    input_val_0 = dataset3.loc[num][0]
    input_val_1 = dataset3.loc[num][1]
    input_val_2 = dataset3.loc[num][2]
    input_val_3 = dataset3.loc[num][3]
    input_val_4 = dataset3.loc[num][4]
    input_val_5 = dataset3.loc[num][5]
    output_val = dataset3.loc[num][6]

    input_np_str = [
        input_val_0,
        input_val_1,
        input_val_2,
        input_val_3,
        input_val_4,
        input_val_5,
    ]
    input_np_float = np.asarray(input_np_str, dtype=np.float64, order="C")
    output_int = int(output_val)

    dataset3_prepared_X.append(input_np_float)
    dataset3_prepared_Y.append(output_int)


### Variables for Neural Network
SLICE_POINT = 1400  # 70% for trainning, 30% for testing
LEARNING_RATE = 0.8
EPOCHS = 800

trainning_data_X = np.array(dataset3_prepared_X[:SLICE_POINT])
trainning_data_Y = np.array(dataset3_prepared_Y[:SLICE_POINT])
test_data_X = np.array(dataset3_prepared_X[SLICE_POINT:])
test_data_Y = np.array(dataset3_prepared_Y[SLICE_POINT:])

n = NeuralNetwork(EPOCHS, trainning_data_X, trainning_data_Y, LEARNING_RATE)
n.train()
n.test(test_data_X, test_data_Y)
n.graph()

input("Press any button to close")
############# DATA SET 3 LOGIC START #############
