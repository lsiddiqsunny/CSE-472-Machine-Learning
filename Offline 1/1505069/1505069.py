#!/usr/bin/env python
# coding: utf-8

# Import libraries for the offline.
# 

# In[1]:


import numpy as np
import pandas as pd
import math
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import OrdinalEncoder
import time
from sklearn.preprocessing import KBinsDiscretizer


# Code for loading telco-customer-churn data

# In[17]:


# https://towardsdatascience.com/preprocessing-with-sklearn-a-complete-and-comprehensive-guide-670cb98fcfb9

def loadTelcoData():
    dataframe = pd.read_csv('content/WA_Fn-UseC_-Telco-Customer-Churn.csv', delimiter=",")


    #removing unnecessary attribute
    dataframe = dataframe.drop(['customerID'], axis=1)
    
    label = 'Churn'
    
    attributes = list(dataframe.columns.values)
    attributes.remove(label)
    
    dataframe[label]= dataframe[label].replace('No', -1)
    dataframe[label]= dataframe[label].replace('Yes', 1)
    
    #removing data with missing label
    dataframe = dataframe.dropna(axis=0, subset=[label])
    dataframe = dataframe.reset_index(drop=True)
    
    non_categorical =['TotalCharges','tenure','MonthlyCharges']
    for column in dataframe:
        dataframe[column] = dataframe[column].replace(r'^\s*$', np.nan, regex=True)
        if dataframe[column].isnull().sum()!=0:
            if column not in non_categorical:
                #use most_frequent data for missing data for non_catagorical data
                imp = SimpleImputer(strategy="most_frequent")
                dataframe[column] = imp.fit_transform(dataframe[column].values.reshape(-1, 1)).reshape(-1)
            else:
                #use mean for missing data for catagorical data
                imp = SimpleImputer(missing_values=np.nan, strategy='mean')
                dataframe[column] = imp.fit_transform(dataframe[column].values.reshape(-1, 1)).reshape(-1)
            print (column,' has missing values')
        if column in non_categorical:
            print (column,'is continuous')
            dataframe[column] = dataframe[column].astype(float)

            # use mean value as threshold
            Threshold = dataframe[column].mean();
            binarizer = Binarizer(threshold=Threshold, copy=True)
            dataframe[column] = pd.Series(binarizer.fit_transform(dataframe[column].values.reshape(-1, 1)).reshape(-1))

    return dataframe,attributes,label


# In[18]:


def loadAdultData():
    attributes = ['age','workclass','fnlwgt','education','education-num',
                  'marital-status','occupation','relationship','race','sex',
                  'capital-gain','capital-loss','hours-per-week','native-country','salary']
    dataframe = pd.read_csv('content/adult.data', delimiter=", ",names=attributes)
    
    label = 'salary'
    
    attributes = list(dataframe.columns.values)
    attributes.remove(label)

    dataframe = dataframe.replace('?', pd.np.nan)
    dataframe[label]= dataframe[label].replace('<=50K', -1)
    dataframe[label]= dataframe[label].replace('>50K', 1)
    
    dataframe = dataframe.dropna(axis=0, subset=[label])
    dataframe = dataframe.reset_index(drop=True)

    #print dataframe.isnull().sum()
    #print (dataframe.shape)
    non_categorical =['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']
    for column in dataframe:
        if dataframe[column].isnull().sum()!=0:
            if column not in non_categorical:
                #use most_frequent data for missing data for non_catagorical data
                imp = SimpleImputer(strategy="most_frequent")
                dataframe[column] = imp.fit_transform(dataframe[column].values.reshape(-1, 1)).reshape(-1)
            else:
                imp = SimpleImputer(missing_values=np.nan, strategy='mean')
                dataframe[column] = imp.fit_transform(dataframe[column].values.reshape(-1, 1)).reshape(-1)
            print (column,' has missing values')
        if column in non_categorical:
            print (column,'is continuous')
            dataframe[column] = dataframe[column].astype(float)

            # use mean value as threshold
            Threshold = dataframe[column].mean();
            binarizer = Binarizer(threshold=Threshold, copy=True)
            dataframe[column] = pd.Series(binarizer.fit_transform(dataframe[column].values.reshape(-1, 1)).reshape(-1))

    return dataframe,attributes,label


# In[19]:


def loadAdultTest():
    attributes = ['age','workclass','fnlwgt','education','education-num',
                  'marital-status','occupation','relationship','race','sex',
                  'capital-gain','capital-loss','hours-per-week','native-country','salary']
    dataframe = pd.read_csv('content/adult.test', delimiter=", ",names=attributes)
    
    label = 'salary'
    
    attributes = list(dataframe.columns.values)
    attributes.remove(label)
    
    dataframe = dataframe.replace('?', pd.np.nan)
    dataframe[label]= dataframe[label].replace('<=50K.', -1)
    dataframe[label]= dataframe[label].replace('>50K.', 1)
    
    dataframe = dataframe.dropna(axis=0, subset=[label])
    dataframe = dataframe.reset_index(drop=True)

    #print dataframe.isnull().sum()
    #print (dataframe.shape)
    non_categorical =['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']
    for column in dataframe:
        if dataframe[column].isnull().sum()!=0:
            if column not in non_categorical:
                #use most_frequent data for missing data for non_catagorical data
                imp = SimpleImputer(strategy="most_frequent")
                dataframe[column] = imp.fit_transform(dataframe[column].values.reshape(-1, 1)).reshape(-1)
            else:
                #use mean for missing data for catagorical data
                imp = SimpleImputer(missing_values=np.nan, strategy='mean')
                dataframe[column] = imp.fit_transform(dataframe[column].values.reshape(-1, 1)).reshape(-1)
            print (column,' has missing values')
            #print (dataframe[column].isnull().sum())

        if column in non_categorical:
            print (column,'is continuous')
            dataframe[column] = dataframe[column].astype(float)

            # use mean value as threshold
            Threshold = dataframe[column].mean();
            binarizer = Binarizer(threshold=Threshold, copy=True)
            dataframe[column] = pd.Series(binarizer.fit_transform(dataframe[column].values.reshape(-1, 1)).reshape(-1))

    return dataframe,attributes,label


# In[20]:


def sampleCreditCardData():
    dataframe = pd.read_csv("content/creditcard.csv", delimiter=",")

    dataframe = dataframe.drop(['Time'], axis=1)
    label = 'Class'
    
    attributes = list(dataframe.columns.values)
    attributes.remove(label)
    
    dataframe = dataframe.dropna(axis=0, subset=[label])
    dataframe = dataframe.reset_index(drop=True)

    
    print (dataframe.shape)


    dataframe_true = dataframe.loc[dataframe[label] == 1]
    dataframe_false = dataframe.loc[dataframe[label] == 0]
    print (dataframe_true.shape)
    print (dataframe_false.shape)

    dataframe_false = dataframe_false.sample(n=20000)
    
    sampleddata = dataframe_true.append(dataframe_false, ignore_index=True)
    sampleddata = sampleddata.sample(frac=1)
    print (sampleddata.shape)
    
    sampleddata.to_csv('content/creditTemp.csv', sep=',',index=False)





def processCreditCardData():
    dataframe = pd.read_csv('content/creditcard.csv', delimiter=",")
    
    dataframe = dataframe.drop(['Time'], axis=1)
    
    label = 'Class'
    attributes = list(dataframe.columns.values)
    attributes.remove(label)

    dataframe = dataframe.dropna(axis=0, subset=[label])
    dataframe = dataframe.reset_index(drop=True)
    
    dataframe[label]= dataframe[label].replace(0, -1)
    dataframe[label]= dataframe[label].replace(1, 1)

    non_categorical = list(dataframe.columns.values)
    non_categorical.remove(label)
    #print(dataframe.shape)
    
    for column in dataframe:
        #print(dataframe[column].isnull().sum())
        if dataframe[column].isnull().sum()!=0:
            #print dataframe[column].values

            if column not in non_categorical:
                #use most_frequent data for missing data for non_catagorical data
                imp = SimpleImputer(strategy="most_frequent")
                dataframe[column] = imp.fit_transform(dataframe[column].values.reshape(-1, 1)).reshape(-1)
            else:
                #use mean for missing data for catagorical data
                imp = SimpleImputer(missing_values=np.nan, strategy='mean')
                dataframe[column] = imp.fit_transform(dataframe[column].values.reshape(-1, 1)).reshape(-1)
            print (column,' has missing values')
            #print (dataframe[column].isnull().sum())

        if column in non_categorical:
            print (column,'is continuous')
            dataframe[column] = dataframe[column].astype(float)

            # use mean value as threshold
            disc = KBinsDiscretizer(n_bins=5, encode='ordinal',strategy='uniform')
            #print("here",disc.fit_transform(dataframe[column].values.reshape(-1, 1)).shape)
            dataframe[column] = pd.Series(disc.fit_transform(dataframe[column].values.reshape(-1, 1)).reshape(-1))

    #dataframe.to_csv('/content/creditTemp.csv', sep=',')


    return dataframe,attributes,label


# Split data set as 80% in train data and 20% in test data.

# In[8]:


def splitDataFrame(dataframe):
    split_at = int(0.8 * dataframe.shape[0])
    
    train_dataframe = dataframe.iloc[:split_at, :]
    test_dataframe = dataframe.iloc[split_at:, :]
    
    train_dataframe = train_dataframe.reset_index(drop=True)
    test_dataframe = test_dataframe.reset_index(drop=True)
    
    return train_dataframe, test_dataframe


# Check which element is most in the data.

# In[38]:


def pluralityValue(data):
    classes = {}
    uniqueValues, occurCount = np.unique(data, return_counts=True)
    for i in range(len(uniqueValues)):
        classes[uniqueValues[i]]=occurCount[i]
    maximum = max(classes, key=classes.get)
    return maximum


# Implementation of entropy.
# 
# 
# 

# In[49]:


def entropy(data):
    classes = {}
    uniqueValues, occurCount = np.unique(data, return_counts=True)
    for i in range(len(uniqueValues)):
        classes[uniqueValues[i]]=occurCount[i]
    size = len(data)
    en = 0.0
    for key in classes:
        prob = (classes[key]*1.0)/size
        en += -1.0*math.log(prob,2.0)*prob
    return en


# In[48]:


def information_gain(data,label):
    parent_entropy = entropy(label)
    split = {}
    size = len(data)
    for i in range(size):
        if data[i] not in split:
            split[data[i]] = []
        split[data[i]].append(label[i])

    child_entropy = 0.0
    for key in split:
        child_size = len(split[key])
        child_entropy += (child_size*1.0*entropy(split[key]))/size

    return parent_entropy-child_entropy


# In[50]:


def Decision_Tree_Learning(𝑒𝑥𝑎𝑚𝑝𝑙𝑒s,parent_𝑒𝑥𝑎𝑚𝑝𝑙𝑒𝑠,attributes,label,mainDataFrame,depth,max_depth = None):
    node = {}
    #print(depth)
    # no more examplse, return plurality value of parent_examples
    if 𝑒𝑥𝑎𝑚𝑝𝑙𝑒𝑠.shape[0] == 0:
        node['leaf'] = True
        node['class'] = pluralityValue(parent_𝑒𝑥𝑎𝑚𝑝𝑙𝑒𝑠[label].values)
        return node
    # No more information gain means entropy is zero, also means all examples have same classification. So return it
    elif entropy(𝑒𝑥𝑎𝑚𝑝𝑙𝑒𝑠[label].values) == 0:
        node['leaf'] = True
        node['class'] = 𝑒𝑥𝑎𝑚𝑝𝑙𝑒𝑠[label].values[0]
        return node
    # attributes is empty or rechead at maximum depty, 
    elif len(attributes)==0 or (max_depth!=None and depth==max_depth):
        node['leaf'] = True
        node['class'] = pluralityValue(𝑒𝑥𝑎𝑚𝑝𝑙𝑒𝑠[label].values)
        return node
    else:
        node['leaf'] = False
        best_info_gain = None
        best_attribute = None
        size = len(attributes)

        for i in range(size):
            info_gain = information_gain(𝑒𝑥𝑎𝑚𝑝𝑙𝑒𝑠[attributes[i]].values,𝑒𝑥𝑎𝑚𝑝𝑙𝑒𝑠[label].values)
            if best_info_gain is None or info_gain>best_info_gain:
                best_info_gain = info_gain
                best_attribute = attributes[i]
        attributes.remove(best_attribute)
        #print(best_attribute)
        node['attribute'] = best_attribute
        values = np.array(mainDataFrame[best_attribute].unique())
        size = len(values)
        #print(values)
        for i in range(size):
            sub_dataframe = 𝑒𝑥𝑎𝑚𝑝𝑙𝑒𝑠.loc[𝑒𝑥𝑎𝑚𝑝𝑙𝑒s[best_attribute] == values[i]]
            sub_dataframe = sub_dataframe.reset_index(drop=True)
            node[values[i]] = Decision_Tree_Learning(sub_dataframe.copy(),𝑒𝑥𝑎𝑚𝑝𝑙𝑒𝑠.copy(),attributes[:],label,mainDataFrame,(depth+1),max_depth)
    return node


# In[54]:


def predict(dataframe,decision_tree,label):
    size = dataframe.shape[0]
    true = dataframe[label].values
    pred = []
    for i in range(size):
        tree = decision_tree
        while not tree['leaf']:
            attribute = tree['attribute']
            featurevalue = dataframe[attribute].values[i]
            tree = tree[featurevalue]
        pred.append(tree['class'])
    return true,pred


# In[62]:


def adaboost(data,booster,attributes,label,mainDataFrame):
    w = []
    z = []
    h = []
    size = data.shape[0]
    for _ in range(size):
        w.append(1)
    w = [(i*1.0) / sum(w) for i in w]
    k = 0

    while not k==booster:
        dataframe = data.copy()
        sampled_frame = dataframe.sample(frac=1,weights=w,replace=True)
        root = Decision_Tree_Learning(sampled_frame,sampled_frame,attributes[:],label,mainDataFrame,0,1)
        error = 0.0
        true,pred = predict(dataframe,root,label)
        for i in range(size):
            if true[i]!=pred[i]:
                error += w[i]
        if error>0.5:
            print ('discarded')
            continue
        h.append(root)
        print (k+1)
        for i in range(size):
            if true[i] == pred[i] and  error!=0:
                w[i] = w[i]*(error/(1.0-error))
        w = [(i*1.0) / sum(w) for i in w]
        if  error!=0:
            weight = (1.0-error)/error
        else:
            weight = float("inf")
        z.append(math.log(weight,2))
        k +=1

    return h,z


# In[66]:


def calculatePerformance(test_Y, pred_Y):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    condition_positive = 0
    condition_negative = 0

    for i in range(len(test_Y)):
        if pred_Y[i] == 1:
            if test_Y[i] == 1:
                true_positive += 1
                condition_positive += 1
            else:
                false_positive += 1
                condition_negative += 1

        else:
            if test_Y[i] == 1:
                false_negative += 1
                condition_positive += 1
            else:
                true_negative += 1
                condition_negative += 1

    print ('True Positive:',true_positive)
    print ('False Positive:',false_positive)
    print ('False Negative:',false_negative)
    print ('True Negative:',true_negative)
    print ('Condition Positive:',condition_positive)
    print ('Condition Negative:',condition_negative)

    
    accuracy = ((true_positive+true_negative)*1.0)/len(test_Y)
    print ('Accuracy:',accuracy*100,'%')
    try:
        recall = (true_positive*1.0)/condition_positive
        true_negative_rate = (true_negative*1.0)/condition_negative
        precision = (true_positive*1.0)/(true_positive+false_positive)
        false_discovery_rate = (false_positive*1.0)/(true_positive+false_positive)
        f1score = 2.0/((1.0/recall)+(1.0/precision))

        
        print ('Recall:',recall)
        print ('Ture Negative rate:',true_negative_rate)
        print ('Precision:',precision)
        print ('False Discovery rate:',false_discovery_rate)
        print ('F1 Score:',f1score)
    except:
        print("division by zero")


# In[68]:


def adaboostPredict(dataframe,h,z,label):
    true_label = dataframe[label].values

    pred_all = []
    size = len(h)
    for i in range(size):
        _,pred = predict(dataframe,h[i],label)
        pred_all.append(pred)

    pred_final = []
    size = dataframe.shape[0]

    for i in range(size):
        value = 0
        for k in range(len(z)):
            value += pred_all[k][i]*z[k]
        if value>=0:
            pred_final.append(1)
        else:
            pred_final.append(-1)
    calculatePerformance(true_label,pred_final)


# In[69]:


dataframe,attributes,label  = loadTelcoData()
train_data,test_data = splitDataFrame(dataframe)
print('Starting trining for decision tree')
start = time.time()
root = Decision_Tree_Learning(train_data.copy(),train_data.copy(),attributes[:],label,dataframe.copy(),0)
end = time.time()
print('Elapsed time: ',end-start)
#print(root)
true,pred = predict(train_data.copy(),root,label)
print ('Training result')
calculatePerformance(true,pred)

true,pred = predict(test_data.copy(),root,label)

print ('Test result')
calculatePerformance(true,pred)

for i in range(1,5):
    k = i*5
    h, z = adaboost(train_data.copy(), k, attributes[:], label,dataframe)
    print ('Round',k)
    print ('Training Dataset')
    adaboostPredict(train_data.copy(),h,z,label)
    print ('Test Dataset')
    adaboostPredict(test_data.copy(), h, z, label)


# In[ ]:


train_data,attributes,label  = loadAdultData()
#print(dataframe[label].head())
#train_data,test_data = splitDataFrame(dataframe)
test_data,attributes,label  = loadAdultTest()
print(train_data.shape)
print(test_data.shape)
#print(label)

print('Starting trining for decision tree')
start = time.time()
root = Decision_Tree_Learning(train_data.copy(),train_data.copy(),attributes[:],label,train_data.copy(),0)
end = time.time()
print('Elapsed time: ',end-start)
print(root)
true,pred = predict(train_data.copy(),root,label)


print ('Training result')
calculatePerformance(true,pred)

true,pred = predict(test_data.copy(),root,label)

print ('Test result')
calculatePerformance(true,pred)
for i in range(1,5):
    k = i*5
    print('Start processing..............')
    h, z = adaboost(train_data.copy(), k, attributes[:], label,dataframe)
    print ('Round',k)
    print ('Training Dataset')
    adaboostPredict(train_data.copy(),h,z,label)
    print ('Test Dataset')
    adaboostPredict(test_data.copy(), h, z, label)


# In[ ]:


#sampleCreditCardData()
dataframe,attributes,label  = processCreditCardData()
#print(dataframe.describe)
#print(dataframe[label].head())
train_data,test_data = splitDataFrame(dataframe.copy())

print('Starting trining for decision tree')
start = time.time()
root = Decision_Tree_Learning(train_data.copy(),train_data.copy(),attributes[:],label,dataframe.copy(),0)
end = time.time()
print('Elapsed time: ',end-start)
print(root)
true,pred = predict(train_data.copy(),root,label)

print ('Training result')
calculatePerformance(true,pred)

true,pred = predict(test_data.copy(),root,label)
print ('Test result')
calculatePerformance(true,pred)

for i in range(1,5):
    k = i*5
    print('Start processing..............')
    h, z = adaboost(train_data.copy(), k, attributes[:], label,dataframe)
    print ('Round',k)
    print ('Training Dataset')
    adaboostPredict(train_data.copy(),h,z,label)
    print ('Test Dataset')
    adaboostPredict(test_data.copy(), h, z, label)

