import numpy as np

x = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.99, 0.01, 0.01, 0.01, 0.01]) # 9 
i = np.argmax(x)

def perceptron(w, b):
    y = np.dot(w, x) + b
    if y >= 0:
        return 1
    else:
        return 0 

b = -1

w1 = np.array([0, 2, 0, 2, 0, 2, 0, 2, 0, 2])
w2 = np.array([0, 0, 2, 2, 0, 0, 2, 2, 0, 0])    
w3 = np.array([0, 0, 0, 0, 2, 2, 2, 2, 0, 0])
w4 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 2, 2])

o1 = perceptron(w1, b)
o2 = perceptron(w2, b)
o3 = perceptron(w3, b)
o4 = perceptron(w4, b)

print('Binary representation of number', i, ':', [o4, o3, o2, o1])

