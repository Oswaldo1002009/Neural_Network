import math
import numpy as np
import random
import copy

#creates a new array with the same size but using zeros
def clone_zeros(l):
	for i in range(len(l)):
		for j in range(len(l[i])):
			for k in range(len(l[i][j])):
				l[i][j][k]=0.0
	return l

#initialice weights witn random numbers
def init_weights(samples, nodes):
	w = [0]*len(nodes)
	for i in range(len(w)):#for each layer
		w[i] = [0]*len(nodes[i])
		for j in range(len(w[i])):#for each node of the current layer
			w[i][j] = []
			if i == 0:
				for k in range(len(samples[0])):
					#w[i][j].append(random.random())
					w[i][j].append(0.5)
			else:
				for k in range(len(nodes[i-1])):
					#w[i][j].append(random.random())
					w[i][j].append(0.5)
	return list(w)

#activation function
def squash(method,net):
	if method == "sigmoid": #logistic
		return 1.0/(1.0 + math.exp(-net))
	if method == "relu":
		return max(net,0)

def calculate_sample_error(outputs, true_outputs):
	sample_error = 0.0
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

#backwards process to calculate deltas, gradients and new weights
def calculate_gradients(sample, weights, nodes, outputs, learning_rate):
	z = np.array(clone_zeros(copy.deepcopy(weights)))
	deltas = copy.deepcopy(z)
	gradients = copy.deepcopy(z)
	new_weights = copy.deepcopy(z)
	for i in range(len(deltas[-1])):#We first start with the last weights
		for j in range(len(deltas[-1][i])):
			out = nodes[-1][i]
			target = outputs[i]
			deltas[-1][i][j] = (out-target)*out*(1-out)
			gradients[-1][i][j] = deltas[-1][i][j]*nodes[-2][j]
			new_weights[-1][i][j] = weights[-1][i][j] - learning_rate*gradients[-1][i][j]

	layers = len(deltas)-1
	for i in range(layers):#All layers except the last, backwards, until layer 0
		for j in range(len(deltas[i])):
			for k in range(len(deltas[i][j])):
				out = nodes[layers-1-i][j]
				t_d = np.array(deltas[layers-i]).T
				t_w = np.array(weights[layers-i]).T
				deltas[layers-1-i][j][k] = np.dot(t_d[j],t_w[j])*out*(1-out)
				if i == 0:#Work with sample values then
					gradients[layers-1-i][j][k] = deltas[layers-1-i][j][k] * sample[k]
				else:
					gradients[layers-1-i][j][k] = deltas[layers-1-i][j][k] * nodes[layers-1-i][k]
				new_weights[layers-1-i][j][k] = weights[layers-1-i][j][k] - learning_rate*gradients[layers-1-i][j][k]
	#return deltas
	#return gradients
	return new_weights

def calculate_error(iteration, samples, outputs, nodes, weights, bias, method):
	total_error = 0.0
	for i in range(len(samples)):
		nodes = calculate_nodes(samples[i], nodes, weights, bias, method)
		#print(nodes)#hyp(x)
		sample_error = calculate_sample_error(nodes[-1],outputs[i])#One set of outputs for each set of samples
		total_error += sample_error
	print("Iteration: %i. Error: %f" %(iteration+1,total_error))
	__errors__.append(total_error)

################################################################################
##############################THE ALGORITHM STARTS##############################
################################################################################

learning_rate = 0.5
__errors__ = []

#Data
#samples = np.array([[1.,1.],[0.,1.],[1.,0.],[0.,0.]]) 
#outputs = np.array([[0.],[1.],[1.],[0.]])
samples = np.array([[.05,.10]])
outputs = np.array([[.01,.99]])


#Forward pass
#nodes = np.array([[0.0, 0.0, 0.0 ,0.0, 0.0],[0.0]])
#weights = np.array(init_weights(samples,nodes))
#bias = np.zeros(len(nodes))+0.5
nodes = np.array([[.0,.0],[.0,.0]])
weights = np.array([[[.15,.20],[.25,.30]],[[.40,.45],[.50,.55]]])
bias = np.array([.35,.60])

#Backward
deltas = np.array([[0.0, 0.0, 0.0 ,0.0, 0.0],[0.0]])

#Method definition
method = "sigmoid"

#Error estimation for the first time, this won't affect future results
calculate_error(-1, samples, outputs, nodes, weights, bias, method)

#Backpropagation
for i in range(10000):
	actual_sample = i%len(samples)
	weights = calculate_gradients(samples[actual_sample], weights, nodes, outputs[actual_sample], learning_rate)
	calculate_error(i, samples, outputs, nodes, weights, bias, method)

import matplotlib.pyplot as plt
#plt.plot(__errors__)
#plt.show()