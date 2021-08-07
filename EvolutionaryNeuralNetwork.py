

import numpy as np
import matplotlib.pyplot as plt
import random

random.seed(17)
np.random.seed(117)

def readData1():
    datafile1 = open('data1.txt')
    datalines1 = datafile1.read().split('\n') # reading lines
    dataio1 = [[0 for x in range(2)]for y in range(32)]
    for i in range(1, 33): # seperationg into inputs and outputs
        dataio1[i-1] = datalines1[i].split(' ')
    datainputs1 = [[0 for x in range(5)]for y in range(32)]
    for i in range(32):
        for j in range(5): # taking inputs
            datainputs1[i][j] = int(dataio1[i][0][j])
    dataoutputs1 = [[0 for x in range(1)] for y in range(32)]
    for i in range(32): # taking outputs
        dataoutputs1[i][0] = int(dataio1[i][1])
    datafile1.close()

    return np.array(datainputs1), np.array(dataoutputs1)

def readData2():
    datafile2 = open('data2.txt')
    datalines2 = datafile2.read().split('\n') # reading lines
    dataio2 = [[0 for x in range(2)]for y in range(64)]
    for i in range(1, 65): # seperationg into inputs and outputs
        dataio2[i-1] = datalines2[i].split(' ')
    datainputs2 = [[0 for x in range(6)]for y in range(64)]
    for i in range(64):
        for j in range(6): # taking inputs
            datainputs2[i][j] = int(dataio2[i][0][j])
    dataoutputs2 = [[0 for x in range(1)] for y in range(64)]
    for i in range(64): # taking outputs
        dataoutputs2[i][0] = int(dataio2[i][1])
    datafile2.close()

    return np.array(datainputs2), np.array(dataoutputs2)

def readData3():
    datafile3 = open('data3.txt')
    datalines3 = datafile3.read().split('\n') # reading lines
    dataio3 = [[0 for x in range(7)]for y in range(1000)]
    for i in range(1, 1001): # seperationg into inputs and outputs
        dataio3[i-1] = datalines3[i].split(' ')
    datainputs3 = [[0 for x in range(6)]for y in range(1000)]
    for i in range(1000):
        for j in range(6): # taking inputs
            datainputs3[i][j] = float(dataio3[0][j])
    dataoutputs3 = [[0 for x in range(1)] for y in range(1000)]
    for i in range(1000): # taking outputs
        dataoutputs3[i][0] = float(dataio3[i][5])
    datafile3.close()

    return np.array(datainputs3), np.array(dataoutputs3)


def sig(x): #sigmoid function
    return 1.0 / (1 + np.exp(-x))

def sig_d(x): # sigmoid derivative function
    return x * (1.0 - x)


class NeuralNetwork:
    def __init__(self, i, o, n = 4, n2 =4, l = 0.3):
        self.hiddenLayer1   = n # number of neurons in the hidden layer
        self.hiddenLayer2   = n2
        self.input          = i
        self.biases         = np.ones((self.input.shape[0], 1)) # setting up biases
        self.input          = np.concatenate((self.input, self.biases), axis = 1)
        self.weights1       = np.random.rand(self.input.shape[1],self.hiddenLayer1) # input weights
        self.weights2       = np.random.rand(self.hiddenLayer1,self.hiddenLayer2) # layer 1 weights
        self.weights3       = np.random.rand(self.hiddenLayer2,1) # layer 2 weights
        self.expected       = o
        self.learningRate   = l
        self.output         = np.zeros(self.expected.shape)
        self.error          = np.array([])

    def feedForward(self):
        self.layer1         = sig(np.dot(self.input, self.weights1))
        self.layer2         = sig(np.dot(self.layer1, self.weights2))
        self.output         = sig(np.dot(self.layer2, self.weights3))
        self.error          = self.expected - self.output

    def backProp(self): #for two layer network
        # back propagation of error using chain rule
        loss           = 2*(self.error) * sig_d(self.output)
        self.weights3 += np.multiply(self.learningRate, np.dot(self.layer2.T, loss))
        loss           = np.dot(loss, self.weights3.T) * sig_d(self.layer2)
        self.weights2 += np.multiply(self.learningRate, np.dot(self.layer1.T, loss))
        loss           = np.dot(loss, self.weights2.T) * sig_d(self.layer1)
        self.weights1 += np.multiply(self.learningRate, np.dot(self.input.T,  loss))

    def assignWeights(self, w1, w2, w3):
        self.weights1 = w1
        self.weights2 = w2
        self.weights3 = w3


class GeneticAlgorithm:
    def __init__(self, c, f, cx = 0.2, mx = 0.07, d = 1):
        self.chromosomes    = c
        self.population     = self.chromosomes.shape[0] # population size
        self.fitness        = f
        self.crossoverX      = cx
        self.mutationX       = mx

    def setChromosomes(self, c):
        self.chromosomes = c

    def setFitness(self, f):
        self.fitness = f

    def rouletteWheel(self):
        temp = self.chromosomes[:]
        for i in range(self.population): # selection for population size
            rouletteWheel = random.uniform(0, np.sum(self.fitness)) # roulette wheel
            cumulativeFitness = 0
            j = 0
            while cumulativeFitness < rouletteWheel: # checking the one selected
                cumulativeFitness += self.fitness[j]
                self.chromosomes[i] = temp[j]
                j += 1

    def crossover(self):
        crossoverPosition = 0
        parent1 = np.array([])
        parent2 = np.array([])
        for i in range(0, int(self.population), 2):
            if random.random() < self.crossoverX:
                crossoverPostion = random.randint(0, self.chromosomes.shape[1])
                parent1 = self.chromosomes[i]
                parent2 = self.chromosomes[i+1]
                self.chromosomes[i] = np.append(parent1[:crossoverPosition], parent2[crossoverPosition:]) # offspring 1
                self.chromosomes[i+1] = np.append(parent2[:crossoverPosition], parent1[crossoverPosition:]) # offsprinf 2

    def mutate(self):
        chromosomeLength = self.chromosomes.shape[1]
        for i in range(self.population):
            if random.random() < self.mutationX:
                self.chromosomes[i][int(random.uniform(0,chromosomeLength))] = random.random() # change a random gene


if __name__ == "__main__":

    inputs, outputs = readData1()

    # Initializing neural network
    inputs              = np.array([[0,0,0,0],
                                    [0,0,0,1],
                                    [0,0,1,0],
                                    [0,0,1,1],
                                    [0,1,0,0],
                                    [0,1,0,1],
                                    [0,1,1,0],
                                    [0,1,1,1],
                                    [1,0,0,0],
                                    [1,0,0,1],
                                    [1,0,1,0],
                                    [1,0,1,1],
                                    [1,1,0,0],
                                    [1,1,0,1],
                                    [1,1,1,0],
                                    [1,1,1,1]]) # for testing
    outputs             = np.array([[0],[1],[1],[0],[1],[1],[1],[0],[0],[0],[1],[0],[1],[1],[1],[0]]) # for testing
    hiddenLayer1Size    = 4
    hiddenLayer2Size    = 2
    learningRate        = 0.2
    nn                  = NeuralNetwork(inputs, outputs, hiddenLayer1Size, hiddenLayer2Size, learningRate)

    #initializing genetic algorithm
    populationSize  = 100
    maxGeneration   = 10000
    chromosomeArray = np.random.rand(populationSize, nn.input.shape[1] * nn.hiddenLayer1 + nn.hiddenLayer1 * nn.hiddenLayer2 + nn.hiddenLayer2)
    fitnessArray    = np.random.rand(populationSize)
    ga              = GeneticAlgorithm(chromosomeArray, fitnessArray)

    mean_error_list = []
    max_error_list  = []
    min_error_list  = []

    for i in range(maxGeneration):
        # error_list = np.array([])
        # for j in range(populationSize):
        #     tempWeights1 = chromosomeArray[j][:nn.input.shape[1]*nn.hiddenLayer1].reshape(nn.input.shape[1], nn.hiddenLayer1)
        #     tempWeights2 = chromosomeArray[j][nn.input.shape[1]*nn.hiddenLayer1:nn.input.shape[1]*nn.hiddenLayer1+nn.hiddenLayer1*nn.hiddenLayer2].reshape(nn.hiddenLayer1, nn.hiddenLayer2)
        #     tempWeights3 = chromosomeArray[j][nn.input.shape[1]*nn.hiddenLayer1+nn.hiddenLayer1*nn.hiddenLayer2:].reshape(nn.hiddenLayer2, 1)
        #     nn.assignWeights(tempWeights1, tempWeights2, tempWeights3)
        #     nn.feedForward()
        #     chromosomeArray[j] = np.append(np.append(nn.weights1.flatten(), nn.weights2.flatten()), nn.weights3)
        #     error_list = np.append(error_list, [np.sum(np.square(nn.error))/nn.input.shape[1]])
        # ga.setChromosomes(chromosomeArray)
        # fitnessArray = np.reciprocal(np.sqrt(error_list))
        # fitnessArray = np.exp(fitnessArray)
        # ga.setFitness(fitnessArray)
        # ga.rouletteWheel()
        # ga.crossover()
        # ga.mutate()
            
        # mean_error_list += [np.average(error_list)]
        # max_error_list += [np.max(error_list)]
        # min_error_list += [np.min(error_list)]
            
        # neuralnet with backpropagation for testing

        nn.feedForward()    
        nn.backProp()

        mean_error_list += [np.sqrt(np.average(np.square(nn.error)))]
        max_error_list  += [np.max(np.abs(nn.error))]
        min_error_list  += [np.min(np.abs(nn.error))]



    # Plotting error graph

    plt.plot(range(maxGeneration), max_error_list, label = 'Max Error')
    plt.plot(range(maxGeneration), mean_error_list, label = 'Mean Error')
    plt.plot(range(maxGeneration), min_error_list, label = 'Min Error')

    plt.xlabel('Generation')
    plt.ylabel('Mean Squared Error')
    plt.title('Error Values Over Generations')
    plt.xlim(left = 0, right = maxGeneration)
    plt.ylim(bottom = 0)
    plt.legend()
    
    plt.show()