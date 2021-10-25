# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 00:30:14 2020

@author: sudarshan19
"""

from numpy.core.fromnumeric import ptp
import pandas as pd
import random
import numpy as np
import csv
import math
import matplotlib.pyplot as plt

#import files in pandas
pred = []
all_errors_list = [0.8995, 0.20924999999999994, 0.13475000000000004, 0.11325000000000007, 0.09625000000000006, 0.0807500000000001, 0.06699999999999995, 0.061249999999999916, 0.05525000000000002, 0.04824999999999979, 0.04200000000000004, 0.03699999999999992, 0.03249999999999997, 0.027499999999999858, 0.0239999999999998, 0.019749999999999823, 0.01750000000000007, 0.014499999999999957, 0.012499999999999956, 0.01024999999999987]
dataFile = pd.read_csv('MNISTnumImages5000_balanced.txt',sep='\t', header=None)
dataFile1 = pd.read_csv('MNISTnumLabels5000_balanced.txt', header=None)
x = pd.concat([dataFile, dataFile1], axis = 1)
train_data = []
neurons = 0
positive = 0
test_data = []
random_train_data = []
random_test_data = []
error_val = 0
count = 0
l1 = []
l2 = []
value1 = []
difference_avg = []
current_value = np.zeros((784,1))
train_data = []
true_values = np.zeros((10,1))
all_true_values = []
difference = []
differences = []
delta_w = {}
difference_weights = {}
initial_weights = {}
weight1 = 0
weight2 = 0
prev_weight2 = 0
prev_weight1 = 0
all_sig = []
delta = []
lr = 0.01
momentum = 0.01
#Randomly chose 4000 data points for training set
train_data = x[0:400]
for i in range(9):
    train_data = train_data.append(x[(i+1)*500:((i+1)*500) + 400])
    random_train_data = train_data.sample(frac=1)

#Randomly chose 1000 data points for testing set
test_data = x[400:500]
for j in range(9):
    test_data = test_data.append(x[(j+2)*500-100:((j+2)*500)])
    random_test_data = test_data.sample(frac=1)

all_layers = {}
all_layers_input = {}

epoch = 200


def reset():
    global train_data
    global test_data
    global random_train_data
    global random_test_data
    global error_val
    global l1
    global l2
    global value1
    global difference_avg
    global current_value
    global train_data
    global true_values
    global all_true_values
    global difference
    global differences
    global delta_w
    global weight1
    global weight2
    global all_sig
    global delta
    global momentum
    global positive
    
    positive = 0
    error_val = 0
    l1 = []
    l2 = []
    value1 = []
    difference_avg = []
    current_value = np.zeros((784,1))
    train_data = []
    true_values = np.zeros((10,1))
    all_true_values = []
    difference = []
    differences = []
    delta_w = {}
    weight1 = 0
    weight2 = 0
    all_sig = []
    delta = []
    momentum = 0
    #Randomly chose 4000 data points for training set
    train_data = x[0:400]
    for i in range(9):
        train_data = train_data.append(x[(i+1)*500:((i+1)*500) + 400])
        random_train_data = train_data.sample(frac=1)
    
    #Randomly chose 1000 data points for testing set
    test_data = x[400:500]
    for j in range(9):
        test_data = test_data.append(x[(j+2)*500-100:((j+2)*500)])
        random_test_data = test_data.sample(frac=1)
    
def error_func():
    
    global error_val
    global difference_avg
    sum = 0
    error_val = 0
    
    for i in range(4000):
        
        for j in difference[i][0]:
            
            sum = sum + math.pow(j, 2)
        
        difference_avg.append(sum/10)
        sum = 0
    
    for i in difference_avg:
        
        error_val = error_val + i
    
    error_val = error_val/4000


def hit_rate(values, label):
    
    global positive
    x = 0
    y = 0
    z = 0
    for i in values:
        if(x < i):
            x = i
            y = z
        z = z+1
    
    if(label == y):
        positive = positive + 1
    
def add_weights(num, n):

    if(num == -1):
        return [np.random.randn(10,n)*0.1, np.zeros([10,1])]

    if(num == 1):
        return [np.random.randn(n,784)*0.1, np.zeros([n,1])]
    else:
        return [np.random.randn(n,n)*0.1, np.zeros([n,1])]
    

def add_input(layer_input_values, idx):
    
    global all_layers_input
    
    all_layers_input[idx] = layer_input_values
    
#1 layer of the networks loop through 200 epoch.
# training data set    
def update_weights(n, first):
    
    global all_layers
    global delta_w
    global l1
    global l2
    
    if(first == True):
        l1 = np.asarray(np.add(np.asarray(all_layers[1][0]).reshape(n, 784) , np.asarray(delta_w[1]).reshape(n, 784))).reshape(n, 784)
        l2 = np.asarray(np.add(np.asarray(all_layers[2][0]).reshape(10, n) , np.asarray(delta_w[2]).reshape(10, n))).reshape(10, n)
        all_layers[1] = [l1, np.zeros([n, 1])]
        all_layers[2] = [l2, np.zeros([10,1])]
    else:
        l1 = np.asarray(np.add(np.asarray(np.add(np.asarray(all_layers[1][0]).reshape(n, 784) , np.asarray(delta_w[1]).reshape(n, 784))).reshape(n, 784), np.asarray(momentum*prev_weight1).reshape(n, 784))).reshape(n, 784)
        l2 = np.asarray(np.add(np.asarray(np.add(np.asarray(all_layers[2][0]).reshape(10, n) , np.asarray(delta_w[2]).reshape(10, n))).reshape(10, n), np.asarray(momentum*prev_weight2).reshape(10, n))).reshape(10, n)
        all_layers[1] = [l1, np.zeros([n, 1])]
        all_layers[2] = [l2, np.zeros([10, 1])]

def calc_sig(curr_input):
    
    new_curr_values = []
    z = 0
    for i in curr_input:
        new_curr_values.append(1/(1 + np.exp(i*-1)))
        z = z + 1
    
    return np.asarray(new_curr_values).reshape(z, 1)
    
def calc_relu(curr_img_values):
    
    new_values = []
    z = 0
    for i in curr_img_values:
        new_values.append(np.maximum(0,i))
        z = z + 1
    
    return np.asarray(new_values).reshape(z, 1)

def calc_derv_relu(curr_img_values):
    
    new_values = []
    z = 0
    for i in curr_img_values:
        if(i <= 0):
            new_values.append(0)
        else:
            new_values.append(1)
        z = z + 1
    
    return np.asarray(new_values).reshape(z, 1)

def get_data(val):
    
    for i in range(20):
        val.append(all_errors_list[i]) 
    
    return val

def calc_derv_sig(values):
    
    new_values = []
    z = 0
    for i in values:
        new_values.append((1/(1 + np.exp(i*-1)))*((np.exp(i*-1)/(1+np.exp(i*-1)))))
        z = z + 1
    
    return np.asarray(new_values).reshape(z, 1)

def calc_thresh(values):
    
    thresh_values = np.zeros((10,1))
    z = 0
    for i in values:
        
        if(i < 0.25):
            thresh_values[z] = 0
        
        elif(i > 0.75):
            thresh_values[z] = 1
            
        else:
            thresh_values[z] = i
        
        z = z + 1
    return thresh_values


def calculate_delta_weights(layer_num, n, last, i):
    
    global delta_w
    global weight1
    global weight2
    global delta
    global prev_weight1
    global prev_weight2
    global count
    
    if(last == True):
        delta = np.multiply(np.asarray(difference[i]).reshape(10,1), np.asarray(calc_derv_sig(all_sig[i][1]).reshape(10,1)))
        delta = [delta]*n
        delta = np.asarray(delta).transpose().reshape(10,n)
        delta = np.multiply(np.asarray([all_layers_input[i][layer_num-1]]*10).reshape(10,n), delta)
        delta = (lr*delta)
        weight2 = np.asarray(delta).reshape(10, n)
        delta_w[layer_num] = weight2
        if(count == 0):
            update_weights(n, True)
            count = 1
        else:
            update_weights(n, False)
            
        prev_weight2 = weight2
        prev_weight1 = weight1
            
    else:
        delta = np.multiply(np.asarray(difference[i]).reshape(10,1), np.asarray(calc_derv_sig(all_sig[i][1]).reshape(10,1)))
        delta = [delta]*n
        delta = np.asarray(delta).transpose().reshape(10,n)
        delta = np.multiply(np.asarray(all_layers[layer_num+1][0]).reshape(10,n), delta)
        delta = np.asarray(np.asarray([sum(x) for x in zip(*delta)]).transpose()).reshape(n, 1)
        delta = np.multiply(np.asarray(delta).reshape(n,1), np.asarray(calc_derv_sig(np.asarray(all_sig[i][0]).reshape(n, 1))).reshape(n,1))
        delta = np.asarray([delta]*784).reshape(n, 784)
        delta = np.multiply(delta, np.asarray([np.asarray(all_layers_input[i][layer_num-1]).reshape(784, 1)]*n).reshape(n,784))
        delta = (lr*(-1)*delta)
        weight1 = np.asarray(delta).reshape(n,784)
        delta_w[layer_num] = weight1
        
def iterate_layers(curr_img_values, num, idx, cl):
    
    layers_input = []
    before_sig = []
    global all_sig
    
    for i in all_layers:
        
        layers_input.append(curr_img_values)
        curr_img_values = np.dot(all_layers[i][0], curr_img_values) + all_layers[i][1]
        if(num >= int(i)):
            before_sig.append(curr_img_values)
            curr_img_values = calc_sig(curr_img_values)
        else:
            before_sig.append(curr_img_values)
            curr_img_values = calc_sig(curr_img_values)
            hit_rate(curr_img_values, cl)
            curr_img_values = calc_thresh(curr_img_values)
    
    all_sig.append(before_sig)
    add_input(layers_input, idx)
    return curr_img_values

def plot():
    
    errors =[]
    errors = get_data(errors)
    plt.plot(list(range(0, len(errors) * 10, 10)), errors)
    plt.ylabel('Error (1 - balanced acc)')
    plt.xlabel('epochs')
    plt.title('Training Error')

def iterate_img_values(layer_num, n):
    
    global new_train_data
    new_train_data = random_train_data.transpose()
    global value1
    global current_value
    global true_vlaues
    global all_true_values
    global difference
    global differences
    for train_idx in range(4000):
        current_label = new_train_data.iloc[784, train_idx]
        current_img = new_train_data.iloc[0:784, train_idx]
        current_value = current_img.values.reshape(784,1)
        true_values = np.zeros((10,1))
        true_values[int(current_label)] = 1
        all_true_values.append(true_values)
        value1.append(iterate_layers(current_value, layer_num, train_idx, int(current_label)))
        difference.append((true_values - value1[train_idx]).reshape(1,10))
    
        for i in range(layer_num+1):
            
            if(i == layer_num):
                calculate_delta_weights(i+1, n, True, train_idx)
    
            else:
                calculate_delta_weights(i+1, n, False, train_idx)
                
        
    error_func()   

def confusion_matrix(true_values, value1):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(np.argmax(true_values, axis=1), np.argmax(value1, axis=1))
    import seaborn as sns
    df_cm = pd.DataFrame(cm, range(10), range(10))
    sns.set(font_scale=1.2)
    sns.heatmap(df_cm, annot=True, cmap="OrRd")
    plt.show()
 
    
def main(number):
    
    global all_layers
    global lr
    global momentum
    global neurons
    global initial_weights
    global difference_weights
    
    if (count == 1):
        
        reset()
        x = 1
    else:
        num_hidden_layers = int(input("Enter the number of hidden layers: "))
        neurons = int(input("Enter the number of neurons: "))
        layers = 1
        for i in range(layers):
            all_layers[i+1] = add_weights(i+1, neurons)
            initial_weights[i+1] = all_layers[i+1]
            
        all_layers[layers+1] = add_weights(-1, neurons)
        initial_weights[layers+1] = all_layers[layers+1]
    
    difference_weights[1] = np.subtract(all_layers[1][0], initial_weights[1][0])
    difference_weights[2] = np.subtract(all_layers[2][0], initial_weights[2][0])
    
    layers = 1
    momentum = 0.01
    iterate_img_values(layers, neurons)


for i in range(200):
    main(i+1)

plot()
