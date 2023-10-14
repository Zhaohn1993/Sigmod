
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier  
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
    
        
        
    filename = 'C:/mood.csv'
    data = pd.read_csv(filename,index_col=False)
    col_name = list(data.columns)
    

            
    x_col = col_name
    col_drop=['id','mood']
   
    for i in col_drop:
        x_col.remove(i)
            
     

    acc=[]       
    X = data[x_col]
    y = data[['mood']]

    

            


    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
          
    best_knn = KNeighborsClassifier	

    pred_X=best_knn.fit(x_train, y_train).predict(x_test)
         


    
    y_true = y_test
    y_pred = pred_X
              
    scores = cross_val_score(best_knn,y_true, y_pred,cv=5,scoring='accuracy')
    
    
    print("accuracy "+"score： ",scores)
    print("accuracy "+"score(mean)： ",scores.mean())
            
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

    
    
    fpr,tpr,threshold = roc_curve(y_true, y_pred) 
    roc_auc = auc(fpr,tpr)
    print("auc： ",roc_auc)

   

   

       
if __name__ == "__main__":       

   classify()
    
    