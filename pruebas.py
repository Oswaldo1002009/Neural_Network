import matplotlib.pyplot as plt
import math
import random
import numpy as np
import copy

samples = np.array([[1.,1.],[0.,1.],[1.,0.],[0.,0.]]) 
y_vec = np.array([[0.],[1.],[1.],[0.]]) 
nodes = np.array([[0.0, 0.0, 0.0 ,0.0, 0.0],[0.0]])
deltas = np.array([[0.0, 0.0, 0.0 ,0.0, 0.0],[0.0]])
#weights = np.array(random_weights(samples,nodes))
#bias = np.array(random_bias(nodes))

x = np.array([[[1.1,1.1],[1.1,1.1],[1.1,1.1],[1.1,1.1],[1.1,1.1]],[[1.1,1.1,1.1,1.1,1.1]]])

def clone_zeros(l):
	for i in range(len(l)):
		for j in range(len(l[i])):
			for k in range(len(l[i][j])):
				l[i][j][k]=0.0
	return l

#print(x)
print(clone_zeros(copy.deepcopy(x)))
print(x)