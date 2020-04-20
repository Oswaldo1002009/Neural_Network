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

a = np.array([1,2])
b = a
print(np.dot(a,b))