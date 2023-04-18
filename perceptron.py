#-------------------------------------------------------------------------
# AUTHOR: Tomes, Christopher
# FILENAME: perceptron.py
# SPECIFICATION: This program builds 2 models using optdigits.tra and optdigits.tes; a Perceptron and a MLPCLassifier. The program will test various hyper_params and print out the ongoing most accurate.
# FOR: CS 4210- Assignment #4
# TIME SPENT: 1.5 hrs
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier #pip install scikit-learn==0.18.rc2 if needed
import numpy as np
import pandas as pd

n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
r = [True, False]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to form the feature data for training
y_training = np.array(df.values)[:,-1]  #getting the last field to form the class label for training

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the data by using Pandas library

X_test = np.array(df.values)[:,:64]    #getting the first 64 fields to form the feature data for test
y_test = np.array(df.values)[:,-1]     #getting the last field to form the class label for test

highest_perceptron_accuracy = 0.0
highest_mlp_accuracy = 0.0

for learning_rate in n: #iterates over n

    for shuffle in r: #iterates over r

        #iterates over both algorithms
        #-->add your Python code here
        b = ['Perceptron','MLP']
        for a in b: #iterates over the algorithms

            #Create a Neural Network classifier
            #if Perceptron then
            #   clf = Perceptron()    #use those hyperparameters: eta0 = learning rate, shuffle = shuffle the training data, max_iter=1000
            #else:
            #   clf = MLPClassifier() #use those hyperparameters: activation='logistic', learning_rate_init = learning rate, hidden_layer_sizes = number of neurons in the ith hidden layer,
            #                          shuffle = shuffle the training data, max_iter=1000
            #-->add your Python code here
            if a == 'Perceptron':
                clf = Perceptron(eta0= learning_rate, shuffle=shuffle, max_iter=1000)
            else:
                h_layer_sizes=(120,)
                clf = MLPClassifier(activation='logistic', learning_rate_init=learning_rate, hidden_layer_sizes=h_layer_sizes, shuffle= shuffle, max_iter=1000)
            #Fit the Neural Network to the training data
            clf.fit(X_training, y_training)
            #make the classifier prediction for each test sample and start computing its accuracy
            #hint: to iterate over two collections simultaneously with zip() Example:
            #for (x_testSample, y_testSample) in zip(X_test, y_test):
            #to make a prediction do: clf.predict([x_testSample])
            #--> add your Python code here
            accuracy = 0.0
            for (x_testSample, y_testSample) in zip(X_test, y_test):
                prediction = clf.predict([x_testSample])
                if prediction == y_testSample:
                    accuracy += 1.0

            #check if the calculated accuracy is higher than the previously one calculated for each classifier. If so, update the highest accuracy
            #and print it together with the network hyperparameters
            #Example: "Highest Perceptron accuracy so far: 0.88, Parameters: learning rate=0.01, shuffle=True"
            #Example: "Highest MLP accuracy so far: 0.90, Parameters: learning rate=0.02, shuffle=False"
            #--> add your Python code here
            accuracy /= len(y_test)

            if a == 'Perceptron' and accuracy > highest_perceptron_accuracy:
                highest_perceptron_accuracy = accuracy
                print(f"Highest Perceptron accuracy so far: {highest_perceptron_accuracy:.2f}, Parameters: learning rate={learning_rate}, shuffle={shuffle}")
            if a == 'MLP' and accuracy > highest_mlp_accuracy:
                highest_mlp_accuracy  = accuracy
                print(f"Highest MLP accuracy so far: {highest_mlp_accuracy:.2f}, Parameters: learning rate={learning_rate}, shuffle={shuffle}, hidden_layer_sizes={h_layer_sizes}")   











