# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 16:10:22 2023

@author: zhaoh
"""


# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 13:41:55 2023

@author: zhaoh
"""


from deepforest import CascadeForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import xlrd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
import pandas as pd

def classify():
    filename = 'C:/Users/zhaoh/Desktop/data/mood/bigfive/high/Neuroticism.csv'
    data = pd.read_csv(filename,index_col=False)
    col_name = list(data.columns)

    x_col = col_name
    """
    col_drop=['id2','mood','Agreeableness','Extraversion','Conscientiousness','Neuroticism','Openness']
    """
    col_drop=['id2','mood']

    for i in col_drop:
        x_col.remove(i)

    acc=[]       
    X = data[x_col]
    y = data[['mood']]

    names=[]
    for j in range (X.shape[1]):
        names.append(j)
    """
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    model = CascadeForestClassifier(random_state=123)
    model.fit(x_train, y_train.values.ravel())
    pred_X = model.predict(x_test)
    """  
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=1)
    model = CascadeForestClassifier(random_state=1)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    y_true = y_test
    scores = accuracy_score(y_test, y_pred)


    print("accuracy "+"score： ",scores)


    accuracy = accuracy_score(y_true, y_pred)
    acc.append(accuracy)

    kappa_value = cohen_kappa_score(y_true, y_pred)

    print("Kappa: ", kappa_value)

    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred, average='micro')
    f1score2 = f1_score(y_true, y_pred, average='weighted')

 
    print("precision： ",p)
    print("recall： ",r)
    print("f1score2： ",f1score2)
    fpr,tpr,threshold = roc_curve(y_true, y_pred) ###计算真正率和假正率
    roc_auc = auc(fpr,tpr) ###计算auc的值
    print("auc： ",roc_auc)
    
if __name__ == "__main__":       
   classify()
