# E8 - XOR Gate Implementation : Neural Networks Perceptron

import numpy as np

def unitStep(val):
    return 1 if val >= 0 else 0

def perceptronModel(x, w, b):
    val = np.dot(x, w) + b
    return unitStep(val)

def NOT_logicGate(x):
    wNOT = -1
    bNOT = 0.5
    return perceptronModel(x, wNOT, bNOT)

def AND_logicGate(x):
    wAND = np.array([1, 1])
    bAND = -1.5
    return perceptronModel(x, wAND, bAND)

def OR_logicGate(x):
    wOR = np.array([1, 1])
    bOR = -0.5
    return perceptronModel(x, wOR, bOR)

def XOR_logicGate(x):
    y1 = AND_logicGate(x)
    y2 = OR_logicGate(x)
    y3 = NOT_logicGate(y1)
    final_x = np.array([y2, y3])
    final_y = AND_logicGate(final_x)
    return final_y

test1 = np.array([0, 0])
test2 = np.array([0, 1])
test3 = np.array([1, 0])
test4 = np.array([1, 1])

print("XOR({}, {}) : {}".format(0, 0, XOR_logicGate(test1)))
print("XOR({}, {}) : {}".format(0, 1, XOR_logicGate(test2)))
print("XOR({}, {}) : {}".format(1, 0, XOR_logicGate(test3)))
print("XOR({}, {}) : {}".format(1, 1, XOR_logicGate(test4)))