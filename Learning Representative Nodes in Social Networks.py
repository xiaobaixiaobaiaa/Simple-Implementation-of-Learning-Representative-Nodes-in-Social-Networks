import numpy as np
from sklearn import preprocessing

# graph = np.array([[0,1,1,1],
#                   [1,0,0,0],
#                   [1,0,0,0],
#                   [1,0,0,0]])

# graph = np.array([[0,1,0,0,0],
#                   [1,0,1,0,0],
#                   [1,1,0,1,1],
#                   [0,0,1,0,1],
#                   [0,0,0,1,0]])

graph = np.array([[0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0]])
graph[0,1], graph[0,2], graph[0,3], graph[0,4] = 1, 1, 1, 1
graph[1,0] = 1
graph[2,0] = 1
graph[3,0] = 1
graph[4,5], graph[4,6], graph[4,7] = 1, 1, 1
graph[5,4] = 1
graph[6,4] = 1
graph[7,4] = 1

a = np.zeros([np.shape(graph)[0], np.shape(graph)[1]])
for i in range(np.shape(graph)[0]):
    for j in range(np.shape(graph)[1]):
        a[i,j] = graph[i,j] / sum(graph[i,:])
# print(a)

v_ = 0.2
# print(v_)

p_j_i = np.zeros([np.shape(graph)[0], np.shape(graph)[1]])
for i in range(np.shape(graph)[0]):
    for j in range(np.shape(graph)[1]):
        if i != j:
            p_j_i[j,i] = (1 - v_)*a[i,j] + (v_ / ((np.shape(graph)[0])-1))
# print(p_j_i)

alpha = np.array([1/len(a) for i in range(np.shape(graph)[0])])
p_0 = v_ / (len(a) - 1)
#print(alpha)

def softmax_(array):
    temp = 0
    for i in range(len(array)):
        temp += (np.exp(array[i]))
    result = np.zeros(len(array))
    for i in range(len(array)):
        result[i] = np.exp(array[i]) / temp
    return result

def min_max_normalization(array):
    max_ = max(array)
    min_ = min(array)
    for i in range(len(array)):
        array[i] = (array[i] - min_) / (max_ - min_)
    return array

learning_rate = 0.1
convergence = 0.000001
errors = 9999
n_epochs = 20000
epoch = 0
while errors >= convergence and epoch < n_epochs:
    nabla = np.zeros(len(a))
    nabla_c = 0

    for i in range(len(a)):
        # for p_j
        temp_1 = 0
        for j in range(len(a)):
            if graph[j,i] == 1:
                temp_1 += (alpha[j] / sum(graph[j]))
        p_i = (1 - v_) * temp_1 + v_ * (1 - alpha[i]) / (len(a) - 1)

        # for e_i
        temp_2 = 0
        for j in range(len(a)):
            if p_j_i[j,i] != 0 and graph[j,i] == 1:
                temp_2 += (alpha[j] * np.log(p_j_i[j,i]) / sum(graph[j]))
        temp_3 = 0
        for j in range(len(a)):
            if p_j_i[j, i] != 0 and i != j:
                temp_3 += (alpha[j] * np.log(p_j_i[j,i]) / (len(a)-1))
        e_i = -(1 - v_) * temp_2 - v_ * temp_3
        e_i /= p_i

        # for nabla
        for j in range(len(a)):
            if graph[j,i] == 1:
                if p_j_i[j, i] != 0:
                    nabla[j] = nabla[j] - (1 - v_) * (np.log(p_j_i[j,i]) + e_i) / (sum(graph[j]) * p_i)

        for j in range(len(a)):
            if graph[i,j] == 1:
                if p_j_i[j, i] != 0:
                    nabla[j] = nabla[j] - v_ * ((np.log(p_j_i[j,i]) - np.log(p_0)) / ((len(a) - 1) * p_i))

        nabla[i] = nabla[i] + v_ * (np.log(p_0) + e_i) / ((len(a) - 1) * p_i)
        nabla_c = nabla_c - v_ * (np.log(p_0) + e_i) / ((len(a) - 1) * p_i)

    nabla = nabla + nabla_c
    alpha_temp = alpha * np.exp(-learning_rate * nabla * alpha)
    errors = sum(abs(alpha - alpha_temp))

    for i in range(len(alpha)):
        alpha[i] = alpha_temp[i] / sum(alpha_temp)
    print(alpha)
    epoch += 1

# print(alpha)
