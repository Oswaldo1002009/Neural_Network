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

print(np.dot([1,2,3],[1,2,3]))
print(np.zeros(3)+1)