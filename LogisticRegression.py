# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 11:17:38 2023

@author: zhaoh
"""


# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 23:54:23 2022

@author: zhaoh
"""


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
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
'''
    n_estimators=10：决策树的个数
    max_features: 选择最适属性时划分的特征不能超过此值
    predict(x)：直接给出预测结果。内部还是调用的predict_proba()，根据概率的结果看哪个类型的预测值最高就是哪个类型。  
   
'''   
"""
def read(file, sheet_index=0):
    
    workbook = xlrd.open_workbook(file)
    sheet = workbook.sheet_by_index(sheet_index)
    print("工作表名称:", sheet.name)
    print("行数:", sheet.nrows)
    print("列数:", sheet.ncols)

    data = []
    for i in range(0, sheet.nrows):
        data.append((sheet.row_values(i)))
    return data
"""

def classify():
    
        
        
    filename = 'C:/Users/zhaoh/Desktop/data/mood/mood2.csv'
    data = pd.read_csv(filename,index_col=False)
    col_name = list(data.columns)#获取所有列名
    

            
    x_col = col_name
    col_drop=['id2','mood']#一些无意义的列，以及标签列'Churn?'
   
    for i in col_drop:
        x_col.remove(i)
            
     
    






#读取数据
    acc=[]       
    X = data[x_col]
    y = data[['mood']]

    

            

#划分训练测试集
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
          
    LR = LogisticRegression()
    LR.fit(x_train,y_train)
    predict_results=LR.predict(x_test)


    
    y_true = y_test
    y_pred = predict_results
         
#训练        

    scores = cross_val_score(LR,x_train,y_train,cv=10,scoring='accuracy') 
    
    
    print("accuracy "+"score： ",scores)
    print("accuracy "+"score(mean)： ",scores.mean())
            
    accuracy = accuracy_score(y_true, y_pred)
    acc.append(accuracy)
        #with open("accuracy_"+str(n)+".txt","w") as f:             
        #f.write(str(accuracy))
              
    #print("accuracy "+"accuracy： ",acc)
    
    
            
    #print(acc)
  
 
#计算Kappa值 
    
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
   """
    x=read('x2.xlsx')   
    np.save('x_data.npy',x)
    np.savetxt('x_data.csv',x)
    y=read('y2.xlsx')   
    np.save('y_data.npy',y)
    np.savetxt('y_data.csv',y)
    print("save data -->")
    """
   classify()
    
    