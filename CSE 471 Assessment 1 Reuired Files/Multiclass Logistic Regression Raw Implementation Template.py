#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Import required python module(s)
import numpy as np 


# In[3]:


#load data from file
data = np.genfromtxt('iris_multiclass.csv', delimiter=',',skip_header=True)

#Distribute data into train and test sets
X_train = data[:120,[0,1,2,3]]
Y_train = data[:120,5]

X_test = data[-30:,[0,1,2,3]]
Y_test = data[-30:,5]
print(Y_train)


# In[4]:


#Define the required Sigmoid function
def sigmoid(z):
    return 1/(1+np.exp(-z))


# In[5]:


#Define the Raw implementation function to set the parameters (theta)

def fit_implementation(X_train, Y_train, learning_rate=0.0005, max_iteration=1000, debug=False):
    #Adding a column of 1's so that the first element of each input is always 1
    #It would be multiplied with theta_0 later
    X_train= np.insert(X_train, 0, values=1, axis=1)
    no_attributes = X_train.shape[1]
    
    #Initialize model parameters theta
    theta = np.zeros((no_attributes,1))
    
    #Run number of iterations
    for icount in range(max_iteration):
        #delta is the quantity that will be added with theta during updating theta
        delta = np.zeros((no_attributes,1))
        totalLogLikelihood = 0
        #Check each data point
        for instance, actualOutput in zip(X_train,Y_train):
            instance=instance.reshape(no_attributes,1)
            dotResult = np.dot(theta.T, instance)
            
            predictedValue=sigmoid(dotResult).squeeze()
            #Calculate the derivative value for this data point
            derivativeValue = instance*(actualOutput-predictedValue)
            #Calculate the amount to be added with theta
            delta += learning_rate*derivativeValue

            logLikelihood = actualOutput*np.log(predictedValue)+(1-actualOutput)*np.log(1-predictedValue)
            totalLogLikelihood += logLikelihood
        theta = theta + delta
        
        #After each 100 iteration, print the status
        if icount%100==0 and debug==True:
            print(icount)
            print(totalLogLikelihood)
            print(theta)
            
    print(totalLogLikelihood)
    print(theta)
    
    return theta


def multciClassFitImplementation(X_train, Y_train):
    #Determine the list unique classes (unique target variable values) 
    #Changes required here
    None

    #For each uniqueclass, determine the best classifier/parameter/theta which best separates the class with others
    #You can temporarily modify Y_train data to achieve the target and can call the fit_implementation function
    parameters = dict()
    #Changes required here
    None
        
    return parameters
        
parameters = multciClassFitImplementation(X_train, Y_train)


# In[9]:


#One of the following parameters of the function is now thetas which is a dictionary containing (targetClass,theta) 
#as (key,value) pairs for all target classes
def prediction(X_test, Y_test, thetas):
    #Adding a column of 1's so that the first element of each input is always 1
    #It would be multiplied with theta_0 later
    X_test= np.insert(X_test, 0, values=1, axis=1)
    no_attributes = X_test.shape[1]
    
    correctCount = 0
    totalCount = 0
    
    maxPredictedValue = -10000
    predictedClass = 1.0
    
    #Check each data point
    for instance, actualOutput in zip(X_test,Y_test):
            instance=instance.reshape(no_attributes,1)
            #Determine the maximum predicted value and predictedClass
            #Changes required here
            None

            print(maxPredictedValue, predictedClass, actualOutput)
            if predictedClass == actualOutput:
                correctCount += 1
            totalCount += 1
    print("Total Correct Count: ",correctCount," Total Wrong Count: ",totalCount-correctCount," Accuracy: ",(correctCount*100)/(totalCount))
    
prediction(X_test, Y_test, parameters)


# # Expected Output: 
# Total Correct Count:  30  Total Wrong Count:  0  Accuracy:  100.0

# In[ ]:




