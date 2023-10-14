from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import xlrd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix 
'''
    n_estimators=10：决策树的个数
    max_features: 选择最适属性时划分的特征不能超过此值
    predict(x)：直接给出预测结果。内部还是调用的predict_proba()，根据概率的结果看哪个类型的预测值最高就是哪个类型。  
   
'''   

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

def classify():
#读取数据
    num=[10,20,30,40,50]
    epoch=20
    
    for n in num:
        acc=[]
        for e in range(epoch):
            
            X=np.load("integrate_data_"+str(n)+".npy")
            y=np.load("y_data.npy")
            
        
#划分训练测试集
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
          
            max_features =4
            if (x_train.shape[1] > max_features):
               rfc = RandomForestClassifier(n_estimators=10, max_features=max_features,
                                     random_state=123)
            else:
                rfc = RandomForestClassifier(random_state=123)
         
#训练        
            pred_X=rfc.fit(x_train, y_train).predict(x_test)

              
#预测准确率  
            
            accuracy = accuracy_score(y_true=y_test, y_pred=pred_X)
            acc.append(accuracy)
            #with open("accuracy_"+str(n)+".txt","w") as f:             
                 #f.write(str(accuracy))
              
            print(str(n)+"%_data_"+"accuracy "+"accuracy： ",accuracy )
    
        np.savetxt("accuracy_"+str(n)+".csv",acc)     
        #print(acc)
    
    
if __name__ == "__main__":
    y=read('y.xlsx')   
    np.save('y_data.npy',y)
    np.savetxt('y_data.csv',y)
    print("save data -->")
    classify()
   
    