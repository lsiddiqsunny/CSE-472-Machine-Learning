import numpy as np
from numpy import genfromtxt
import pandas as pd
import math



def categoricalReplacingValues(missing_data,label_data):
    bestValues = {}
    for i in range(len(label_data)):
        if not pd.isnull(missing_data[i]):
            if label_data[i] not in bestValues:
                bestValues[label_data[i]] = {}

            set = bestValues[label_data[i]]
            if missing_data[i] not in set:
                set[missing_data[i]] = 0
            set[missing_data[i]] += 1

            bestValues[label_data[i]] = set

    for key in bestValues:
        maximum = max(bestValues[key], key=bestValues[key].get)
        bestValues[key] = maximum

    for i in range(len(label_data)):
        if pd.isnull(missing_data[i]):
            missing_data[i] = bestValues[label_data[i]]

    return missing_data


def continuousReplacingValues(missing_data,label_data):
    bestValues = {}
    for i in range(len(label_data)):
        if not pd.isnull(missing_data[i]):
            if label_data[i] not in bestValues:
                bestValues[label_data[i]] = []
            bestValues[label_data[i]].append(missing_data[i])

    for key in bestValues:
        mean = np.mean(bestValues[key])
        bestValues[key] = mean

    for i in range(len(label_data)):
        if pd.isnull(missing_data[i]):
            missing_data[i] = bestValues[label_data[i]]
    return missing_data


def entropy(data):
    classes = {}
    size = len(data)
    for i in range(size):
        if data[i] not in classes:
            classes[data[i]] = 0
        classes[data[i]] += 1

    en = 0.0
    for key in classes:
        probability = (classes[key]*1.0)/size
        en += -1.0*math.log(probability,2.0)*probability
    return en


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


def binarization(data,label):
    zipped = zip(data.copy(),label.copy())
    sorted_data = [x for x,_ in sorted(zipped)]
    sorted_label = [x for _,x in sorted(zipped)]
    last_data = sorted_data[0]-5.1
    size = len(data)
    best_threshold = None
    best_info_gain = None
    bin_array = []
    for i in range(size):
        bin_array.append(1)

    for i in range(size+1):
        if(i==size):
            threshold = last_data+5.1
        else:
            threshold = (last_data+sorted_data[i])/2.0
            last_data = sorted_data[i]
        if(i>0):
            bin_array[i-1] = 0
        info_gain = information_gain(bin_array,sorted_label)
        if best_info_gain is None or info_gain>best_info_gain:
            best_info_gain = info_gain
            best_threshold = threshold

    for i in range(size):
        if data[i]<=best_threshold:
            data[i] = 0
        else:
            data[i] = 1

    return data

#data = [70,90,85,60,75,220,95,125,100,120]
#label = ['NO','YES','YES','NO','NO','NO','YES','NO','NO','NO']
#print data
#print binarization(data,label)


def shuffle(df, n=1, axis=0):
    df = df.copy()
    for _ in range(n):
        df.apply(np.random.shuffle, axis=axis)
    return df


def loadTelcoData():
    dataframe = pd.read_csv('Telco.csv', delimiter=",")
    non_categorical =['tenure','MonthlyCharges','TotalCharges']
    #non_categorical = []
    label = 'Churn'

    #removing unnecessary attribute
    dataframe = dataframe.drop(['customerID'], axis=1)

    #print dataframe

    #removing entries with no label
    #dataframe[label] = dataframe[label].str.strip()
    #dataframe[label] = dataframe[label].replace("", pd.np.nan, regex=True)
    dataframe = dataframe.dropna(axis=0, subset=[label])
    dataframe = dataframe.reset_index(drop=True)

    #print dataframe.shape

    for column in dataframe:
        #dataframe[column] = dataframe[column].replace("\s*", pd.np.nan, regex=True)
        if dataframe[column].isnull().sum():
            #print dataframe[column].values
            if column not in non_categorical:
                dataframe[column] = \
                    pd.Series(categoricalReplacingValues(dataframe[column].values,dataframe[label].values))
            else:
                dataframe[column] = \
                    continuousReplacingValues(dataframe[column].values,dataframe[label].values)
            print column,' has missing values'
            #print dataframe[column].isnull().sum()

        if column in non_categorical:
            print column,'is continuous'
            dataframe[column] = \
                pd.Series(binarization(dataframe[column].values,dataframe[label].values))

    #dataframe.to_csv('fixed.csv', sep=',')
    #dataframe = dataframe.sample(frac=1)
    #dataframe = dataframe.reindex(np.random.permutation(dataframe.index))
    #dataframe = dataframe.reset_index(drop=True)
    split_point = int(0.8*dataframe.shape[0])
    train_dataframe= dataframe.iloc[:split_point,:]
    test_dataframe = dataframe.iloc[split_point:,:]
    train_dataframe = train_dataframe.reset_index(drop=True)
    test_dataframe = test_dataframe.reset_index(drop=True)
    attributes = list(dataframe.columns.values)
    attributes.remove(label)
    return train_dataframe,test_dataframe,attributes,label


def loadAdultData(open_file,save_file):
    attributes = ['age','workclass','fnlwgt','education','education-num',
                  'marital-status','occupation','relationship','race','sex',
                  'capital-gain','capital-loss','hours-per-week','native-country','salary']
    dataframe = pd.read_csv(open_file, delimiter=", ",names=attributes,engine='python')
    non_categorical =['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']
    #non_categorical = []
    label = 'salary'
    dataframe = dataframe.replace('?', pd.np.nan)
    dataframe[label]= dataframe[label].replace('<=50K', 'No')
    dataframe[label]= dataframe[label].replace('>50K', 'Yes')
    dataframe = dataframe.dropna(axis=0, subset=[label])
    dataframe = dataframe.reset_index(drop=True)
    #print dataframe.isnull().sum()
    print dataframe.shape

    for column in dataframe:
        if dataframe[column].isnull().sum():
            #print dataframe[column].values
            if column not in non_categorical:
                dataframe[column] = \
                    pd.Series(categoricalReplacingValues(dataframe[column].values,dataframe[label].values))
            else:
                dataframe[column] = \
                    continuousReplacingValues(dataframe[column].values,dataframe[label].values)
            print column,' has missing values'
            #print dataframe[column].isnull().sum()

        if column in non_categorical:
            print column,'is continuous'
            dataframe[column] = \
                pd.Series(binarization(dataframe[column].values,dataframe[label].values))
            #print dataframe[column].values

    #dataframe.to_csv(save_file, sep=',',index=False)

    attributes.remove(label)
    print dataframe.shape
    #print attributes
    return dataframe,attributes,label


def sampleCreditCardData():
    dataframe = pd.read_csv("creditcard.csv", delimiter=",",engine='python')
    label = 'Class'
    dataframe = dataframe.dropna(axis=0, subset=[label])
    dataframe = dataframe.reset_index(drop=True)
    dataframe = dataframe.drop(['Time'], axis=1)
    print dataframe.shape
    attributes = list(dataframe.columns.values)
    attributes.remove(label)
    dataframe_true = dataframe.loc[dataframe[label] == 1]
    dataframe_false = dataframe.loc[dataframe[label] == 0]
    print dataframe_true.shape
    print dataframe_false.shape
    dataframe_false = dataframe_false.sample(n=20000)
    bigdata = dataframe_true.append(dataframe_false, ignore_index=True)
    bigdata = bigdata.sample(frac=1)
    print bigdata.shape
    bigdata.to_csv('creditTemp.csv', sep=',', index=False)




def processCreditCardData():
    dataframe = pd.read_csv("creditTemp.csv", delimiter=",", engine='python')
    label = 'Class'
    non_categorical = list(dataframe.columns.values)
    non_categorical.remove(label)

    for column in dataframe:
        if dataframe[column].isnull().sum():
            #print dataframe[column].values
            if column not in non_categorical:
                dataframe[column] = \
                    pd.Series(categoricalReplacingValues(dataframe[column].values,dataframe[label].values))
            else:
                dataframe[column] = \
                    continuousReplacingValues(dataframe[column].values,dataframe[label].values)
            print column,' has missing values'
            #print dataframe[column].isnull().sum()

        if column in non_categorical:
            print column,'is continuous'
            dataframe[column] = \
                pd.Series(binarization(dataframe[column].values,dataframe[label].values))
            #print dataframe[column].values

    dataframe.to_csv("creditcard_fixed.csv", sep=',',index=False)
    return 0


def loadDataGeneric(filename,label):
    dataframe = pd.read_csv(filename, delimiter=",")
    #print dataframe.shape
    #print dataframe.isnull().sum()
    attributes = list(dataframe.columns.values)
    attributes.remove(label)
    #print attributes
    return dataframe,attributes,label


def loadPlayData():
    dataframe = pd.read_csv('play.csv', delimiter=",")
    non_categorical = []
    label = 'Play'

    dataframe = dataframe.dropna(axis=0, subset=[label])
    dataframe = dataframe.reset_index(drop=True)

    #print dataframe.shape

    for column in dataframe:
        # dataframe[column] = dataframe[column].replace("\s*", pd.np.nan, regex=True)
        if dataframe[column].isnull().sum():
            # print dataframe[column].values
            if column not in non_categorical:
                dataframe[column] = \
                    pd.Series(categoricalReplacingValues(dataframe[column].values, dataframe[label].values))
            else:
                dataframe[column] = \
                    continuousReplacingValues(dataframe[column].values, dataframe[label].values)
            print column, ' has missing values'
            print dataframe[column].isnull().sum()

        if column in non_categorical:
            print column, 'is continuous'
            dataframe[column] = \
                pd.Series(binarization(dataframe[column].values, dataframe[label].values))

    attributes = list(dataframe.columns.values)
    attributes.remove(label)
    #print attributes
    #print dataframe
    return dataframe,attributes,label


def processBankData():
    dataframe = pd.read_csv('banknote.txt', delimiter=",")
    non_categorical =['A','B','C','D']
    #non_categorical = []
    label = 'E'
    #print dataframe

    for column in dataframe:
        #dataframe[column] = dataframe[column].replace("\s*", pd.np.nan, regex=True)
        if dataframe[column].isnull().sum():
            #print dataframe[column].values
            if column not in non_categorical:
                dataframe[column] = \
                    pd.Series(categoricalReplacingValues(dataframe[column].values,dataframe[label].values))
            else:
                dataframe[column] = \
                    continuousReplacingValues(dataframe[column].values,dataframe[label].values)
            print column,' has missing values'
            #print dataframe[column].isnull().sum()

        if column in non_categorical:
            print column,'is continuous'
            dataframe[column] = \
                pd.Series(binarization(dataframe[column].values,dataframe[label].values))

    dataframe = dataframe.sample(frac=1)
    dataframe.to_csv('fixed_banknote.csv', sep=',', index=False)



def splitDataFrame(dataframe):
    split_point = int(0.8 * dataframe.shape[0])
    train_dataframe = dataframe.iloc[:split_point, :]
    test_dataframe = dataframe.iloc[split_point:, :]
    train_dataframe = train_dataframe.reset_index(drop=True)
    test_dataframe = test_dataframe.reset_index(drop=True)
    return train_dataframe, test_dataframe


def pluralityValue(data):
    classes = {}
    size = len(data)
    for i in range(size):
        if data[i] not in classes:
            classes[data[i]] = 0
        classes[data[i]] += 1

    maximum = max(classes, key=classes.get)
    return maximum


def getValues(data):
    values = []
    size = len(data)
    for i in range(size):
        if data[i] not in values:
            values.append(data[i])

    return values



def buildDecisionTree(dataframe,parent_dataframe,attributes,depth,max_depth,label,mainDataFrame):
    node = {}
    if dataframe.shape[0] == 0:
        node['leaf'] = True
        node['class'] = pluralityValue(parent_dataframe[label].values)
        return node
    elif entropy(dataframe[label].values) == 0:
        node['leaf'] = True
        node['class'] = dataframe[label].values[0]
        return node
    elif len(attributes)==0 or depth==max_depth:
        node['leaf'] = True
        node['class'] = pluralityValue(dataframe[label].values)
        return node
    else:
        node['leaf'] = False
        size = len(attributes)
        best_info_gain = None
        best_attribute = None
        for i in range(size):
            info_gain = information_gain(dataframe[attributes[i]].values,dataframe[label].values)
            if best_info_gain is None or info_gain>best_info_gain:
                best_info_gain = info_gain
                best_attribute = attributes[i]
        attributes.remove(best_attribute)
        node['attribute'] = best_attribute
        values = getValues(mainDataFrame[best_attribute].values)
        size = len(values)
        for i in range(size):
            sub_dataframe = dataframe.loc[dataframe[best_attribute] == values[i]]
            sub_dataframe = sub_dataframe.reset_index(drop=True)
            node[values[i]] = buildDecisionTree(sub_dataframe.copy(),dataframe.copy(),attributes[:],(depth+1),max_depth,label,mainDataFrame)
    return node


def predict(dataframe,decision_tree,label):
    size = dataframe.shape[0]
    true_label = dataframe[label].values
    pred_label = []
    tree = decision_tree
    for i in range(size):
        tree = decision_tree
        while not tree['leaf']:
            attribute = tree['attribute']
            featurevalue = dataframe[attribute].values[i]
            tree = tree[featurevalue]
        pred_label.append(tree['class'])
    return true_label,pred_label



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

    print 'True Positive:',true_positive
    print 'False Positive:',false_positive
    print 'False Negative:',false_negative
    print 'True Negative:',true_negative
    print 'Condition Positive:',condition_positive
    print 'Condition Negative:',condition_negative

    accuracy = ((true_positive+true_negative)*1.0)/len(test_Y)
    recall = (true_positive*1.0)/condition_positive
    true_negative_rate = (true_negative*1.0)/condition_negative
    precision = (true_positive*1.0)/(true_positive+false_positive)
    false_discovery_rate = (false_positive*1.0)/(true_positive+false_positive)
    f1score = 2.0/((1.0/recall)+(1.0/precision))

    print 'Accuracy:',accuracy*100,'%'
    print 'Recall:',recall
    print 'Ture Negative rate:',true_negative_rate
    print 'Precision:',precision
    print 'False Discovery rate:',false_discovery_rate
    print 'F1 Score:',f1score
    #print(classification_report(test_Y,pred_Y))


def encode(true,pred):
    size = len(true)
    for i in range(size):
        if true[i] == 'Yes':
            true[i] = 1
        else:
            true[i] = -1
        if pred[i] == 'Yes':
            pred[i] = 1
        else:
            pred[i] = -1
    return true,pred


def encode1d(data):
    size = len(data)
    for i in range(size):
        if data[i] == 'Yes':
            data[i] = 1
        else:
            data[i] = -1
    return data


def encodeMinus1d(data):
    size = len(data)
    for i in range(size):
        if data[i] == 0:
            data[i] = -1
    return data


def encodeMinus(true,pred):
    size = len(true)
    for i in range(size):
        if true[i] == 0:
            true[i] = -1
        if pred[i] == 0:
            pred[i] = -1
    return true, pred


#dataframe,attributes,label = loadPlayData()
#label_encoding = False
#print dataframe
#root = buildDecisionTree(dataframe,dataframe,attributes,0,10,label,dataframe)
#true,pred = predict(dataframe,root,label)
#calculatePerformance(true,pred)
#print dataframe


def adaboost(data,booster,attributes,label):
    w = []
    z = []
    h = []
    size = data.shape[0]
    for _ in range(size):
        w.append(1)
    w = [float(i) / sum(w) for i in w]
    k = 0

    discarded = 0

    while not k==booster:
        dataframe = data.copy()
        sampled_frame = dataframe.sample(frac=1,weights=w,replace=True)
        root = buildDecisionTree(sampled_frame,sampled_frame,attributes[:],0,1,label,data)
        error = 0.0
        true,pred = predict(dataframe,root,label)
        for i in range(size):
            if not true[i]==pred[i]:
                error += w[i]
        if error>0.5:
            #discarded += 1
            #print 'discarded'
            continue
        h.append(root)
        print (k+1)
        discarded = 0
        for i in range(size):
            if true[i] == pred[i] and not error==0:
                w[i] = w[i]*(error/(1.0-error))
        w = [float(i) / sum(w) for i in w]
        if not error==0:
            weight = (1.0-error)/error
        else:
            weight = float("inf")
        z.append(math.log(weight,2))
        k +=1

    return h,z


def adaboostPredict(dataframe,h,z,label,label_encoding):
    true_label = dataframe[label].values
    if label_encoding:
        true_label = encode1d(true_label)
    else:
        true_label = encodeMinus1d(true_label)

    pred_all = []
    size = len(h)
    for i in range(size):
        _,pred = predict(dataframe,h[i],label)
        if label_encoding:
            pred = encode1d(pred)
        else:
            pred = encodeMinus1d(pred)
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

    #print true_label
    #print pred_final
    count = 0
    for i in range(size):
        if pred_final[i] == true_label[i]:
            count += 1

    accuracy = (count*100.0)/size
    print 'Accuracy:',accuracy,'%'



#train_dataframe, test_dataframe, attributes, label = loadTelcoData()
#dataframe,attributes,label = loadDataGeneric('fixed_credit.csv','Class')
#dataframe,attributes,label = loadDataGeneric('fixed_telco.csv','Churn')
#dataframe,attributes,label = loadDataGeneric('fixed_banknote.csv','E')
#train_dataframe,test_dataframe = splitDataFrame(dataframe)
label_encoding = False #true for telco

train_dataframe,attributes,label = loadDataGeneric('fixed_adult.csv','salary')
test_dataframe,_,_ =loadDataGeneric('fixed_adult_test.csv','salary')


k = 20
h, z = adaboost(train_dataframe.copy(), k, attributes[:], label)
print 'Round',k
print 'Training Dataset'
adaboostPredict(train_dataframe.copy(),h,z,label,label_encoding)
print 'Test Dataset'
adaboostPredict(test_dataframe.copy(), h, z, label, label_encoding)
print '\n'


'''

root = buildDecisionTree(train_dataframe.copy(),train_dataframe.copy(),attributes[:],0,35,label,train_dataframe.copy())
true,pred = predict(train_dataframe.copy(),root,label)

if label_encoding:
    true,pred = encode(true,pred)
else:
    true,pred = encodeMinus(true,pred)
print 'Training result'
calculatePerformance(true,pred)

true,pred = predict(test_dataframe.copy(),root,label)
if label_encoding:
    true,pred = encode(true,pred)
else:
    true,pred = encodeMinus(true,pred)
print 'Test result'
calculatePerformance(true,pred)
'''