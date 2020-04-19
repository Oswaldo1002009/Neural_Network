import math
import numpy as np
import random

#initialice weights witn random numbers
def init_weights(samples, nodes):
	w = [0]*len(nodes)
	for i in range(len(w)):#for each layer
		w[i] = [0]*len(nodes[i])
		for j in range(len(w[i])):#for each node of the current layer
			w[i][j] = []
			if i == 0:
				for k in range(len(samples[0])):
					w[i][j].append(random.random())
			else:
				for k in range(len(nodes[i-1])):
					w[i][j].append(random.random())
	return w

#activation function
def squash(method,net):
	if method == "sigmoid": #logistic
		return 1.0/(1.0 + math.exp(-net))
	if method == "relu":
		return max(net,0)

def calculate_sample_error(outputs, true_outputs):
	sample_error = 0
	for i in range(len(outputs)):
		sample_error += ((outputs[i] - true_outputs[i])**2)/2.0
	return(sample_error)

def calculate_nodes(samples_i, nodes, weights, bias, method):
	for i in range(len(nodes)):
		for j in range(len(nodes[i])):
			if i == 0:  #if it's the first time, it needs to work with inputs in samples_i
				net = np.dot(samples_i, weights[i][j]) + bias[i]
			else: #otherwise, it need to work with nodes of the previous layer
				net = np.dot(nodes[i-1], weights[i][j]) + bias[i]
			nodes[i][j] = squash(method, net)
	return nodes

def calculate_new_weights(learning_rate, samples, outputs, nodes, weights, bias, deltas, method):
	total_error = 0
	for i in range(len(samples)):
		nodes = calculate_nodes(samples[i], nodes, weights, bias, method)
		sample_error = calculate_sample_error(nodes[-1],outputs[i])#One set of outputs for each set of samples
		total_error += sample_error
	__errors__.append(total_error)

learning_rate = 0.5
__errors__ = []

#Data
samples = np.array([[1.,1.],[0.,1.],[1.,0.],[0.,0.]]) 
outputs = np.array([[0.],[1.],[1.],[0.]])

#Forward pass
nodes = np.array([[0.0, 0.0, 0.0 ,0.0, 0.0],[0.0]])
weights = np.array(init_weights(samples,nodes))
bias = np.zeros(len(nodes))+0.5

#Backward
deltas = np.array([[0.0, 0.0, 0.0 ,0.0, 0.0],[0.0]])

#Method definition
method = "sigmoid"


for i in range(10):
	calculate_new_weights(learning_rate, samples, outputs, nodes, weights, bias, deltas, method)

import matplotlib.pyplot as plt
plt.plot(__errors__)
plt.show()